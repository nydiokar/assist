"""Evidence-sufficiency verdict from the top-1 relevance score (v1 §6.2 abstention, F16)."""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.config import AbstentionConfig
from ..domain.models import EvidenceVerdict, RetrievedResult


def assess_evidence(
    results: Sequence[RetrievedResult], cfg: AbstentionConfig
) -> tuple[EvidenceVerdict, list[str]]:
    """Decide evidence sufficiency from the top-1 relevance score.

    Pure function, no LLM. Reads the configured `signal` from the top result's `score_details`;
    `sufficient = top_score >= min_top_score`. Returns the verdict plus any warnings (e.g. the
    reranker is off so only the uncalibrated fusion score is available — v1 §4.6). The caller
    decides whether to suppress passages.
    """
    warnings: list[str] = []
    if not cfg.enabled:
        return EvidenceVerdict(enabled=False, sufficient=True), warnings
    top_score = results[0].score_details.get(cfg.signal) if results else None
    if cfg.signal == "rerank_score" and top_score is None and results:
        warnings.append("abstention: rerank_score absent (reranker off); evidence gate inactive")
        return EvidenceVerdict(enabled=False, sufficient=True, signal=cfg.signal), warnings
    sufficient = top_score is not None and top_score >= cfg.min_top_score
    return (
        EvidenceVerdict(
            enabled=True,
            sufficient=sufficient,
            signal=cfg.signal,
            top_score=top_score,
            threshold=cfg.min_top_score,
        ),
        warnings,
    )
