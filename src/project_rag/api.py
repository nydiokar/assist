"""Public tool contracts (v1 §3.2). The single entrypoint AI-Team binds to at WP4.

ProjectRAG wires config → embedder/stores/manifest/parsers and exposes:
  ingest_project, search_project, open_source, get_ingestion_status.
Project root paths are held server-side per project (never taken from the LLM) to honor the
path-safety and project-scope rules (v1 §8). Query text is bounded and validated.
"""

from __future__ import annotations

from pathlib import Path

from .config import AbstentionConfig, Config, load_config
from .embeddings import build_embedder
from .indexing import IndexingService
from .lexical_store import LexicalStore
from .manifest import Manifest
from .parsers import ParserRegistry
from .retrieval import Retriever
from .schema import (
    EvidenceVerdict,
    IngestReport,
    RetrievedResult,
    SearchResponse,
    SourceLocation,
    SourceResponse,
    StatusReport,
)
from .sources import open_source as _open_source
from .trace import QueryTrace, new_query_id, query_hash, timer


def assess_evidence(results: list[RetrievedResult],
                    cfg: AbstentionConfig) -> tuple[EvidenceVerdict, list[str]]:
    """Decide evidence sufficiency from the top-1 relevance score (v1 §6.2 abstention, F16).

    Pure function, no LLM. Reads the configured `signal` from the top result's `score_details`;
    `sufficient = top_score >= min_top_score`. Returns the verdict plus any warnings (e.g. the
    reranker is off so only the uncalibrated fusion score is available — v1 §4.6). The caller
    decides whether to suppress passages; this function never mutates `results`.
    """
    warnings: list[str] = []
    if not cfg.enabled:
        return EvidenceVerdict(enabled=False, sufficient=True), warnings
    top_score = results[0].score_details.get(cfg.signal) if results else None
    if cfg.signal == "rerank_score" and top_score is None and results:
        warnings.append("abstention: rerank_score absent (reranker off); evidence gate inactive")
        return EvidenceVerdict(enabled=False, sufficient=True, signal=cfg.signal), warnings
    sufficient = top_score is not None and top_score >= cfg.min_top_score
    return (EvidenceVerdict(enabled=True, sufficient=sufficient, signal=cfg.signal,
                            top_score=top_score, threshold=cfg.min_top_score),
            warnings)


class ProjectRAG:
    def __init__(self, config: Config | None = None, config_path: str | None = None) -> None:
        self._cfg = config or load_config(config_path)
        self._embedder = build_embedder(self._cfg.embedding)
        # Vector store built lazily/eagerly here; memory/chroma both via protocol.
        from .vector_store import build_vector_store  # noqa: PLC0415

        self._vs = build_vector_store(self._cfg.vector_store)
        self._lx = LexicalStore(self._cfg.lexical_store.path)
        self._manifest = Manifest(self._cfg.lexical_store.path)
        self._registry = ParserRegistry(self._cfg.parsing)
        cache_root = str(Path(self._cfg.lexical_store.path).resolve().parent / "cache")
        self._indexer = IndexingService(self._cfg, self._embedder, self._vs, self._lx,
                                        self._manifest, self._registry, cache_root=cache_root)
        self._retriever = Retriever(self._embedder, self._vs, self._lx, self._cfg.retrieval)
        self._roots: dict[str, str] = {}      # project_id -> server-side root
        self._last_traces: list[QueryTrace] = []

    @property
    def config(self) -> Config:
        return self._cfg

    def register_project_root(self, project_id: str, root_path: str) -> None:
        """Server-side binding of project scope (v1 §4.6: never trust client project_id alone)."""
        self._roots[project_id] = str(Path(root_path).resolve())

    # ---- tool contracts ----

    def ingest_project(self, project_id: str, root_path: str, *,
                       dry_run: bool = False) -> IngestReport:
        self.register_project_root(project_id, root_path)
        return self._indexer.ingest(project_id, root_path, dry_run=dry_run)

    def search_project(self, project_id: str, query: str, *, top_k: int | None = None,
                       filters: dict | None = None, as_of: str | None = None) -> SearchResponse:
        warnings: list[str] = []
        if len(query) > self._cfg.security.max_query_chars:
            query = query[: self._cfg.security.max_query_chars]
            warnings.append("query truncated to max_query_chars")
        if as_of is not None:
            warnings.append("as_of ignored: source-effective dates not established (v1 §4.8)")
        doc_class: str | None = None
        channels: tuple[str, ...] = ("dense", "lexical")
        for key, val in (filters or {}).items():
            if key == "doc_class" and val in {"project_record", "reference", "filesystem"}:
                doc_class = val
            elif key == "channels" and val and set(val) <= {"dense", "lexical"}:
                channels = tuple(val)  # evaluation baselines (v1 §6.1 B/C/D)
            else:
                warnings.append(f"unsupported filter ignored: {key}")

        gen = self._manifest.active_generation(project_id)
        qid = new_query_id(project_id, query, nonce=gen or "none")
        if gen is None:
            self._last_traces.append(QueryTrace(query_id=qid, tool="search_project",
                                                project_id=project_id,
                                                query_hash=query_hash(query),
                                                index_generation="", result_count=0,
                                                warnings=["no active generation"]))
            return SearchResponse(query_id=qid, project_id=project_id, index_generation="",
                                  results=[], warnings=warnings + ["no active generation"])

        t = timer()
        with t.span("retrieve"):
            results, channels = self._retriever.retrieve(
                project_id, gen, query, self._cfg.security.max_result_excerpt_chars,
                doc_class=doc_class, channels=channels)
        if top_k is not None:
            results = results[:top_k]

        # Evidence-sufficiency gate (v1 §6.2 abstention, F16). Computed from the top-1 relevance
        # score; the verdict is always attached. If configured, insufficient evidence suppresses
        # the passages so a downstream LLM gets an explicit no-evidence signal (v1 §5.2) rather
        # than plausible-but-irrelevant chunks to confabulate over.
        verdict, ev_warnings = assess_evidence(results, self._cfg.retrieval.abstention)
        warnings.extend(ev_warnings)
        served = results
        if verdict.enabled and not verdict.sufficient \
                and self._cfg.retrieval.abstention.suppress_on_insufficient:
            served = []
            warnings.append("insufficient evidence: top relevance "
                            f"{verdict.top_score:.3f} < {verdict.threshold:.3f}; passages withheld")

        # Parent-section context expansion (v1 §7, F12/F9 chunk-granularity gap). Query-time,
        # additive: widen the evidence window around each served chunk so a WP4 answerer can
        # recover answers the top chunk alone did not carry (measured span@5 0.718 vs 0.667).
        # Only on served (non-suppressed) passages; the verbatim excerpt is kept.
        if served and self._cfg.retrieval.context_expansion.enabled:
            from .expand import attach_expanded_context  # noqa: PLC0415

            attach_expanded_context(self, project_id, served,
                                    max_chars=self._cfg.retrieval.context_expansion.max_chars)

        trace = QueryTrace(query_id=qid, tool="search_project", project_id=project_id,
                           query_hash=query_hash(query), index_generation=gen,
                           retrieved_chunk_ids=[r.chunk_id for r in served],
                           channels=channels, result_count=len(served),
                           timings_ms=t.timings_ms, warnings=warnings)
        self._last_traces.append(trace)
        return SearchResponse(query_id=qid, project_id=project_id, index_generation=gen,
                              results=served, evidence=verdict, warnings=warnings)

    def open_source(self, project_id: str, document_id: str, *,
                    source_path: str | None = None, location: SourceLocation | None = None,
                    max_chars: int | None = None) -> SourceResponse:
        root = self._roots.get(project_id)
        if root is None:
            return SourceResponse(project_id=project_id, document_id=document_id,
                                  source_path=source_path or "", location=location, excerpt="",
                                  warnings=["project root not registered server-side"])
        if source_path is None:
            return SourceResponse(project_id=project_id, document_id=document_id,
                                  source_path="", location=location, excerpt="",
                                  warnings=["source_path required to resolve document"])
        return _open_source(project_id=project_id, document_id=document_id, root_path=root,
                            source_path=source_path, registry=self._registry, location=location,
                            max_chars=max_chars or self._cfg.retrieval.max_source_chars)

    def get_ingestion_status(self, project_id: str) -> StatusReport:
        return self._indexer.status(project_id)

    # ---- diagnostics (grounding gate support, v1 §5.2) ----
    @property
    def traces(self) -> list[QueryTrace]:
        return self._last_traces

    def close(self) -> None:
        self._lx.close()
        self._manifest.close()
