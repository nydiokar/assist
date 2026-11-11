"""Fused ranking → served results, and reranker score application. Pure."""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.config import RetrievalConfig
from ..domain.models import ChannelHit, RetrievedResult
from .cap import apply_per_file_cap
from .fusion import FusedCandidate, rrf_fuse


def _to_result(c: FusedCandidate, rank: int) -> RetrievedResult:
    return RetrievedResult(
        chunk_id=c.chunk_id,
        document_id=c.payload.document_id,
        source_path=c.payload.source_path,
        source_location=c.payload.location,
        excerpt=c.payload.citation_text,
        rank=rank,
        retrieval_channels=sorted(c.ranks),
        score_details={
            "fusion_score": round(c.fused, 6),
            **{f"rank_{ch}": float(r) for ch, r in c.ranks.items()},
        },
    )


def fuse_and_rank(
    dense: Sequence[ChannelHit],
    lexical: Sequence[ChannelHit],
    cfg: RetrievalConfig,
    limit: int | None = None,
) -> list[RetrievedResult]:
    """Fuse channels by RRF and apply the per-file cap, returning up to `limit` results.

    `limit` defaults to `cfg.top_k`. When a reranker is active the caller passes a larger limit
    so the reranker reorders the whole candidate pool before the final top_k cut. `per_file_cap
    = None` falls back to the legacy `max(1, limit // 2)`.
    """
    window = limit if limit is not None else cfg.top_k
    ranking = rrf_fuse({"dense": dense, "lexical": lexical}, cfg.fusion.k)
    per_file = cfg.per_file_cap if cfg.per_file_cap is not None else max(1, window // 2)
    picked = apply_per_file_cap(ranking, per_file, window)
    return [_to_result(c, i) for i, c in enumerate(picked, start=1)]


def apply_rerank_scores(
    results: Sequence[RetrievedResult], scores: Sequence[float], top_n: int
) -> list[RetrievedResult]:
    """Reorder the first `top_n` results by descending score (tie-break on chunk_id).

    `scores[i]` belongs to `results[i]` for i < top_n. Results beyond `top_n` keep their order
    behind the reranked head, so pool recall never shrinks. Ranks are renumbered 1..n and each
    head result records `rerank_score` for audit.
    """
    head = list(results[:top_n])
    tail = list(results[top_n:])
    order = sorted(range(len(head)), key=lambda i: (-float(scores[i]), head[i].chunk_id))
    reranked = [
        head[i].model_copy(
            update={
                "score_details": {
                    **head[i].score_details,
                    "rerank_score": round(float(scores[i]), 6),
                }
            }
        )
        for i in order
    ]
    return [r.model_copy(update={"rank": i}) for i, r in enumerate(reranked + tail, start=1)]


def truncate_excerpts(results: Sequence[RetrievedResult], limit: int) -> list[RetrievedResult]:
    return [
        r.model_copy(update={"excerpt": r.excerpt[:limit] + "…"}) if len(r.excerpt) > limit else r
        for r in results
    ]
