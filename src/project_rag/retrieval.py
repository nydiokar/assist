"""Query pipeline: dense + lexical channels, RRF fusion, file-diversity dedup (v1 §4.7/§5.1).

No LLM. Project scope is enforced inside each store before fusion. Deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RetrievalConfig
from .embeddings import EmbedderProtocol
from .lexical_store import LexicalStore
from .schema import RetrievedResult, SourceLocation
from .vector_store import VectorStoreProtocol


@dataclass
class _Candidate:
    chunk_id: str
    payload: dict
    channels: set[str] = field(default_factory=set)
    ranks: dict[str, int] = field(default_factory=dict)


def _rrf(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


def fuse_and_rank(
    dense: list[tuple[str, float, dict]],
    lexical: list[tuple[str, float, dict]],
    cfg: RetrievalConfig,
    limit: int | None = None,
) -> list[RetrievedResult]:
    """Fuse channels by RRF and apply file-diversity dedup, returning up to `limit` results.

    `limit` defaults to `cfg.top_k`. When a reranker is active the caller passes a larger limit
    so the reranker reorders the whole candidate pool before the final top_k cut (otherwise the
    reranker would only ever see the 8 the dedup already picked). The file-diversity cap scales
    with `limit` so it does not over-dedup a large rerank window.
    """
    cap = limit if limit is not None else cfg.top_k
    cands: dict[str, _Candidate] = {}

    for rank, (cid, _score, payload) in enumerate(dense, start=1):
        c = cands.setdefault(cid, _Candidate(cid, payload))
        c.channels.add("dense")
        c.ranks["dense"] = rank
    for rank, (cid, _score, payload) in enumerate(lexical, start=1):
        c = cands.setdefault(cid, _Candidate(cid, payload))
        c.channels.add("lexical")
        c.ranks["lexical"] = rank
        if not c.payload:
            c.payload = payload

    scored: list[tuple[float, _Candidate]] = []
    for c in cands.values():
        fused = sum(_rrf(r, cfg.fusion.k) for r in c.ranks.values())
        scored.append((fused, c))
    scored.sort(key=lambda t: (-t[0], t[1].chunk_id))  # stable deterministic tie-break

    # File-diversity dedup: at most `per_file` chunks from one source in the window, so sibling
    # chunks of the right doc cannot crowd out its evidence chunk (WP3 2026-10-27; evidence@5
    # peaks at per_file=3). Over-cap chunks go to overflow and only refill if the window would
    # otherwise be short — so we never return fewer than `cap` when candidates exist.
    per_file = cfg.per_file_cap if cfg.per_file_cap is not None else max(1, cap // 2)
    results: list[RetrievedResult] = []
    seen_files: dict[str, int] = {}
    overflow: list[tuple[float, _Candidate]] = []
    for fused, c in scored:
        path = c.payload.get("source_path", "")
        if seen_files.get(path, 0) >= per_file:
            overflow.append((fused, c))
            continue
        _append(results, c, fused)
        seen_files[path] = seen_files.get(path, 0) + 1
        if len(results) >= cap:
            break
    for fused, c in overflow:
        if len(results) >= cap:
            break
        _append(results, c, fused)

    return [r.model_copy(update={"rank": i}) for i, r in enumerate(results[:cap], start=1)]


def _append(results: list[RetrievedResult], c: _Candidate, fused: float) -> None:
    loc = c.payload.get("location") or SourceLocation()
    results.append(
        RetrievedResult(
            chunk_id=c.chunk_id,
            document_id=c.payload.get("document_id", ""),
            source_path=c.payload.get("source_path", ""),
            source_location=loc,
            excerpt=c.payload.get("citation_text", ""),
            rank=0,
            retrieval_channels=sorted(c.channels),
            score_details={
                "fusion_score": round(fused, 6),
                **{f"rank_{ch}": float(r) for ch, r in c.ranks.items()},
            },
        )
    )


class Retriever:
    def __init__(
        self,
        embedder: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
        lexical_store: LexicalStore,
        cfg: RetrievalConfig,
    ) -> None:
        self._embedder = embedder
        self._vs = vector_store
        self._lx = lexical_store
        self._cfg = cfg
        self._reranker = None
        if cfg.rerank.enabled:
            from .rerank import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(cfg.rerank)

    def retrieve(
        self,
        project_id: str,
        generation: str,
        query: str,
        excerpt_limit: int,
        *,
        doc_class: str | None = None,
        channels: tuple[str, ...] = ("dense", "lexical"),
    ) -> tuple[list[RetrievedResult], dict[str, int]]:
        dense: list[tuple[str, float, dict]] = []
        lexical: list[tuple[str, float, dict]] = []
        # Cross-language query expansion (F11/F14): append counterpart-language domain anchors so
        # a short BG table row is reachable from an EN query. Additive; same expanded text feeds
        # both channels. Original query is unchanged upstream (trace hashes the original).
        search_query = query
        if self._cfg.query_expansion.enabled:
            from .query_expansion import expand_query

            search_query = expand_query(
                query, corpus_script=self._cfg.query_expansion.corpus_script
            )
        if "dense" in channels:
            qvec = self._embedder.embed_queries([search_query])[0]
            dense = self._vs.query(
                project_id,
                generation,
                np.asarray(qvec),
                self._cfg.candidates_per_channel,
                doc_class,
            )
        if "lexical" in channels:
            lexical = self._lx.search(
                project_id, generation, search_query, self._cfg.candidates_per_channel, doc_class
            )
        if self._reranker is not None:
            # Fuse a wider window, rerank it by content, then cut to top_k.
            pool = fuse_and_rank(dense, lexical, self._cfg, limit=self._cfg.rerank.top_n)
            pool = self._reranker.rerank(query, pool)
            results = pool[: self._cfg.top_k]
        else:
            results = fuse_and_rank(dense, lexical, self._cfg)
        results = [
            r.model_copy(update={"excerpt": r.excerpt[:excerpt_limit] + "…"})
            if len(r.excerpt) > excerpt_limit
            else r
            for r in results
        ]
        channels = {"dense": len(dense), "lexical": len(lexical)}
        return results, channels
