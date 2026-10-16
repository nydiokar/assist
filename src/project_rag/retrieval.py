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
) -> list[RetrievedResult]:
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

    # File-diversity dedup: cap to top_k, prefer spreading across source files.
    results: list[RetrievedResult] = []
    seen_files: dict[str, int] = {}
    overflow: list[tuple[float, _Candidate]] = []
    for fused, c in scored:
        path = c.payload.get("source_path", "")
        if seen_files.get(path, 0) >= max(1, cfg.top_k // 2) and len(scored) > cfg.top_k:
            overflow.append((fused, c))
            continue
        _append(results, c, fused)
        seen_files[path] = seen_files.get(path, 0) + 1
        if len(results) >= cfg.top_k:
            break
    for fused, c in overflow:
        if len(results) >= cfg.top_k:
            break
        _append(results, c, fused)

    for i, r in enumerate(results, start=1):
        r.rank = i
    return results[: cfg.top_k]


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
            score_details={"fusion_score": round(fused, 6),
                           **{f"rank_{ch}": float(r) for ch, r in c.ranks.items()}},
        )
    )


class Retriever:
    def __init__(self, embedder: EmbedderProtocol, vector_store: VectorStoreProtocol,
                 lexical_store: LexicalStore, cfg: RetrievalConfig) -> None:
        self._embedder = embedder
        self._vs = vector_store
        self._lx = lexical_store
        self._cfg = cfg

    def retrieve(self, project_id: str, generation: str, query: str,
                 excerpt_limit: int) -> tuple[list[RetrievedResult], dict[str, int]]:
        qvec = self._embedder.embed_queries([query])[0]
        dense = self._vs.query(project_id, generation, np.asarray(qvec),
                               self._cfg.candidates_per_channel)
        lexical = self._lx.search(project_id, generation, query,
                                  self._cfg.candidates_per_channel)
        results = fuse_and_rank(dense, lexical, self._cfg)
        for r in results:
            if len(r.excerpt) > excerpt_limit:
                r.excerpt = r.excerpt[:excerpt_limit] + "…"
        channels = {"dense": len(dense), "lexical": len(lexical)}
        return results, channels
