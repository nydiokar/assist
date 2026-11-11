"""Search service: hybrid retrieval, evidence gate, context expansion, validated source opens.

Query pipeline (v1 §4.7/§5.1): optional cross-language query expansion → dense + lexical channels
(project + generation scoped inside each store) → RRF fusion + per-file cap → optional reranker
over a wider pool → excerpt bound → evidence-sufficiency verdict (optional suppression) →
optional parent-section expansion. No LLM. Deterministic.

Project roots are held server-side (v1 §4.6/§8): open_source never takes a path from the caller
without resolving it inside the registered root.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import PurePath
from typing import cast

import numpy as np

from ..core.abstention import assess_evidence
from ..core.expand import merge_expansion, parent_location
from ..core.fingerprint import new_query_id, query_hash
from ..core.query_expansion import expand_query
from ..core.ranking import apply_rerank_scores, fuse_and_rank, truncate_excerpts
from ..domain.config import Config
from ..domain.errors import (
    PathOutsideRoot,
    ProviderContractViolation,
    ProviderTimeout,
    ProviderUnavailable,
)
from ..domain.models import (
    ChannelHit,
    EvidenceVerdict,
    FeedbackSignal,
    ProjectHit,
    QueryTrace,
    RetrievedResult,
    SearchAllResponse,
    SearchResponse,
    SourceLocation,
    SourceResponse,
)
from ..ports.catalog import Catalog
from ..ports.clock import Clock
from ..ports.corpus import CorpusReader
from ..ports.embedder import Embedder
from ..ports.feedback import FeedbackSink
from ..ports.lexical import LexicalIndex
from ..ports.parser import DocumentParser
from ..ports.reranker import Reranker
from ..ports.telemetry import Telemetry
from ..ports.vector import VectorIndex

_DOC_CLASSES = frozenset({"project_record", "reference", "filesystem"})
_CHANNELS = frozenset({"dense", "lexical"})


class _Timer:
    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = round((time.perf_counter() - start) * 1000, 3)


def _matches(block_loc: SourceLocation, want: SourceLocation) -> bool:
    if want.page is not None and block_loc.page != want.page:
        return False
    if want.sheet is not None and block_loc.sheet != want.sheet:
        return False
    return not (want.heading is not None and block_loc.heading != want.heading)


class SearchService:
    def __init__(
        self,
        cfg: Config,
        *,
        embedder: Embedder,
        reranker: Reranker | None,
        vector: VectorIndex,
        lexical: LexicalIndex,
        catalog: Catalog,
        corpus: CorpusReader,
        parser: DocumentParser,
        telemetry: Telemetry | None = None,
        feedback: FeedbackSink | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._cfg = cfg
        self._telemetry = telemetry
        self._feedback = feedback
        self._clock = clock
        self._embedder = embedder
        self._reranker = reranker
        self._vector = vector
        self._lexical = lexical
        self._catalog = catalog
        self._corpus = corpus
        self._parser = parser
        self._roots: dict[str, str] = {}  # project_id -> server-side root
        self._traces: list[QueryTrace] = []

    @property
    def traces(self) -> list[QueryTrace]:
        return self._traces

    def register_project_root(self, project_id: str, root_path: str) -> None:
        """Server-side binding of project scope (v1 §4.6: never trust a client project_id alone).

        The root is stored canonicalised (resolving "." inside it yields the root itself)."""
        self._roots[project_id] = self._corpus.resolve(root_path, ".")

    # ---- retrieval ----

    def retrieve(
        self,
        project_id: str,
        generation: str,
        query: str,
        excerpt_limit: int,
        *,
        doc_class: str | None = None,
        channels: Sequence[str] = ("dense", "lexical"),
    ) -> tuple[list[RetrievedResult], dict[str, int], list[str]]:
        """Results, per-channel candidate counts, and degraded components (spec §6.6): a model
        server fault drops that stage and flags it; it never turns into an empty result."""
        rcfg = self._cfg.retrieval
        dense: list[ChannelHit] = []
        lexical: list[ChannelHit] = []
        degraded: list[str] = []
        # Cross-language query expansion (F11/F14): additive anchors for both channels; the trace
        # still hashes the original query.
        search_query = query
        if rcfg.query_expansion.enabled:
            search_query = expand_query(query, corpus_script=rcfg.query_expansion.corpus_script)
        if "dense" in channels:
            try:
                with self._stage("embed"):
                    qmat = self._embedder.embed_queries([search_query])
                qvec = np.asarray(qmat[0], dtype=np.float32)
            except _PROVIDER_ERRORS:  # degrade, do not fail (spec §6.6)
                degraded.append("dense")
            else:
                with self._stage("dense"):
                    dense = self._vector.query(
                        project_id, generation, qvec, rcfg.candidates_per_channel, doc_class
                    )
        if "lexical" in channels:
            with self._stage("lexical"):
                lexical = self._lexical.search(
                    project_id, generation, search_query, rcfg.candidates_per_channel, doc_class
                )
        if self._reranker is not None:
            # Fuse a wider window, rerank it by content, then cut to top_k.
            with self._stage("fuse"):
                pool = fuse_and_rank(dense, lexical, rcfg, limit=rcfg.rerank.top_n)
            if pool:
                head = pool[: rcfg.rerank.top_n]
                try:
                    with self._stage("rerank"):
                        scores = self._reranker.score(query, [r.excerpt or "" for r in head])
                except _PROVIDER_ERRORS:  # keep the RRF order, flag it (spec §6.6)
                    degraded.append("rerank")
                else:
                    pool = apply_rerank_scores(pool, scores, rcfg.rerank.top_n)
            results = (
                fuse_and_rank(dense, lexical, rcfg) if "rerank" in degraded else pool[: rcfg.top_k]
            )
        else:
            with self._stage("fuse"):
                results = fuse_and_rank(dense, lexical, rcfg)
        results = truncate_excerpts(results, excerpt_limit)
        return results, {"dense": len(dense), "lexical": len(lexical)}, degraded

    def search_project(
        self,
        project_id: str,
        query: str,
        *,
        top_k: int | None = None,
        filters: Mapping[str, object] | None = None,
        as_of: str | None = None,
    ) -> SearchResponse:
        sec = self._cfg.security
        warnings: list[str] = []
        if len(query) > sec.max_query_chars:
            query = query[: sec.max_query_chars]
            warnings.append("query truncated to max_query_chars")
        if as_of is not None:
            warnings.append("as_of ignored: source-effective dates not established (v1 §4.8)")
        doc_class: str | None = None
        channels: tuple[str, ...] = ("dense", "lexical")
        for key, val in (filters or {}).items():
            requested = (
                [str(v) for v in cast("Iterable[object]", val)]
                if isinstance(val, list | tuple | set | frozenset)
                else []
            )
            if key == "doc_class" and isinstance(val, str) and val in _DOC_CLASSES:
                doc_class = val
            elif key == "channels" and requested and set(requested) <= _CHANNELS:
                channels = tuple(requested)  # evaluation baselines (v1 §6.1 B/C/D)
            else:
                warnings.append(f"unsupported filter ignored: {key}")

        gen = self._catalog.active_generation(project_id)
        qid = new_query_id(project_id, query, nonce=gen or "none")
        if gen is None:
            self._traces.append(
                QueryTrace(
                    query_id=qid,
                    tool="search_project",
                    project_id=project_id,
                    query_hash=query_hash(query),
                    index_generation="",
                    result_count=0,
                    warnings=["no active generation"],
                )
            )
            return SearchResponse(
                query_id=qid,
                project_id=project_id,
                index_generation="",
                results=[],
                warnings=[*warnings, "no active generation"],
            )

        t = _Timer()
        with t.span("retrieve"):
            results, channel_counts, degraded = self.retrieve(
                project_id,
                gen,
                query,
                sec.max_result_excerpt_chars,
                doc_class=doc_class,
                channels=channels,
            )
        if top_k is not None:
            results = results[:top_k]

        # Evidence-sufficiency gate (v1 §6.2, F16): the verdict is always attached; when
        # configured, insufficient evidence suppresses the passages (explicit no-evidence signal).
        acfg = self._cfg.retrieval.abstention
        verdict, ev_warnings = assess_evidence(results, acfg)
        if "rerank" in degraded:
            # The calibrated signal is gone; the gate result (if any) is RRF-based (spec §6.6).
            verdict = verdict.model_copy(update={"calibrated": False})
        warnings.extend(ev_warnings)
        if degraded:
            warnings.append(f"degraded: {', '.join(degraded)} unavailable; reduced-mode search")
        served = results
        if verdict.enabled and not verdict.sufficient and acfg.suppress_on_insufficient:
            served = []
            warnings.append(
                "insufficient evidence: top relevance "
                f"{verdict.top_score:.3f} < {verdict.threshold:.3f}; passages withheld"
            )

        # Parent-section context expansion (v1 §7, F12/F9): additive, served passages only.
        xcfg = self._cfg.retrieval.context_expansion
        if served and xcfg.enabled:
            t_expand = time.perf_counter()
            served = [
                r.model_copy(
                    update={
                        "expanded_context": self.expand_result_context(
                            project_id, r, max_chars=xcfg.max_chars
                        )
                    }
                )
                for r in served
            ]
            self._observe("expand", time.perf_counter() - t_expand)

        self._record_outcome(
            project_id, query, served, verdict.enabled and not verdict.sufficient, degraded
        )
        self._traces.append(
            QueryTrace(
                query_id=qid,
                tool="search_project",
                project_id=project_id,
                query_hash=query_hash(query),
                index_generation=gen,
                retrieved_chunk_ids=[r.chunk_id for r in served],
                channels=channel_counts,
                result_count=len(served),
                timings_ms=t.timings_ms,
                warnings=warnings,
            )
        )
        return SearchResponse(
            query_id=qid,
            project_id=project_id,
            index_generation=gen,
            results=served,
            evidence=verdict,
            warnings=warnings,
            degraded=degraded,
        )

    def search_all(
        self,
        query: str,
        project_ids: Sequence[str],
        *,
        top_k: int | None = None,
        filters: Mapping[str, object] | None = None,
    ) -> SearchAllResponse:
        """D-12: search each project, then merge by reranker score (fusion score when there is
        no reranker or it was down), keeping `project_id` on every result."""
        hits: list[ProjectHit] = []
        verdicts: dict[str, EvidenceVerdict] = {}
        warnings: list[str] = []
        degraded: set[str] = set()
        for pid in project_ids:
            resp = self.search_project(pid, query, top_k=top_k, filters=filters)
            verdicts[pid] = resp.evidence
            degraded |= set(resp.degraded)
            warnings.extend(f"{pid}: {w}" for w in resp.warnings)
            hits.extend(ProjectHit(project_id=pid, result=r) for r in resp.results)
        hits.sort(key=_merge_key)
        limit = top_k or self._cfg.retrieval.top_k
        return SearchAllResponse(
            results=[
                h.model_copy(update={"result": h.result.model_copy(update={"rank": i})})
                for i, h in enumerate(hits[:limit], start=1)
            ],
            evidence=verdicts,
            warnings=warnings,
            degraded=sorted(degraded),
        )

    # ---- telemetry + automatic feedback (spec §6.9/§6.10) ----

    @contextmanager
    def _stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._observe(name, time.perf_counter() - t0)

    def _observe(self, stage: str, seconds: float) -> None:
        if self._telemetry is not None:
            self._telemetry.observe(stage, seconds)

    def _record_outcome(
        self,
        project_id: str,
        query: str,
        served: Sequence[RetrievedResult],
        abstained: bool,
        degraded: Sequence[str],
    ) -> None:
        outcome = "abstain" if abstained else ("empty" if not served else "answered")
        if self._telemetry is not None:
            self._telemetry.count("assist_search_total", {"outcome": outcome})
            for component in degraded:
                self._telemetry.count("assist_degraded_total", {"component": component})
        if self._feedback is None or self._clock is None:
            return
        kinds = [outcome] if outcome != "answered" else []
        kinds += ["degraded"] if degraded else []
        for kind in kinds:
            self._feedback.record(
                FeedbackSignal(
                    kind=kind,
                    project_id=project_id,
                    at=self._clock.now(),
                    query_hash=query_hash(query),
                    detail=",".join(degraded) if kind == "degraded" else "",
                    query=query,
                )
            )
            if self._telemetry is not None:
                self._telemetry.count("assist_feedback_total", {"kind": kind})

    def expand_result_context(
        self, project_id: str, result: RetrievedResult, *, max_chars: int = 4000
    ) -> str:
        """Parent-section text for one result (its excerpt when there is no coarser anchor)."""
        parent = parent_location(result.source_location)
        if parent is None:
            return result.excerpt
        resp = self.open_source(
            project_id,
            result.document_id,
            source_path=result.source_path,
            location=parent,
            max_chars=max_chars,
        )
        return merge_expansion(resp.excerpt, result.excerpt, max_chars)

    # ---- validated source opens (v1 §5.1 step 7, §8) ----

    def open_source(
        self,
        project_id: str,
        document_id: str,
        *,
        source_path: str | None = None,
        location: SourceLocation | None = None,
        max_chars: int | None = None,
    ) -> SourceResponse:
        root = self._roots.get(project_id)
        if root is None:
            return SourceResponse(
                project_id=project_id,
                document_id=document_id,
                source_path=source_path or "",
                location=location,
                excerpt="",
                warnings=["project root not registered server-side"],
            )
        if source_path is None:
            return SourceResponse(
                project_id=project_id,
                document_id=document_id,
                source_path="",
                location=location,
                excerpt="",
                warnings=["source_path required to resolve document"],
            )
        limit = max_chars or self._cfg.retrieval.max_source_chars
        try:
            abs_path = self._corpus.resolve(root, source_path)
        except PathOutsideRoot as e:
            return SourceResponse(
                project_id=project_id,
                document_id=document_id,
                source_path=source_path,
                location=location,
                excerpt="",
                warnings=[str(e)],
            )
        if not self._corpus.is_file(abs_path):
            return SourceResponse(
                project_id=project_id,
                document_id=document_id,
                source_path=source_path,
                location=location,
                excerpt="",
                warnings=["source file not found"],
            )
        ext = PurePath(abs_path).suffix.lower()
        doc = self._parser.parse(abs_path, source_path, ext)
        blocks = doc.blocks
        if location is not None:
            filtered = [b for b in blocks if _matches(b.location, location)]
            if filtered:
                blocks = filtered
        text = "\n".join(b.text for b in blocks)
        return SourceResponse(
            project_id=project_id,
            document_id=document_id,
            source_path=source_path,
            location=location,
            excerpt=text[:limit],
            truncated=len(text) > limit,
            warnings=list(doc.diagnostics),
        )


_PROVIDER_ERRORS = (ProviderUnavailable, ProviderTimeout, ProviderContractViolation)


def _merge_key(hit: ProjectHit) -> tuple[float, str, str]:
    scores = hit.result.score_details
    primary = scores.get("rerank_score", scores.get("fusion_score", 0.0))
    return (-primary, hit.project_id, hit.result.chunk_id)
