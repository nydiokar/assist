"""Evidence-sufficiency gate (v1 §6.2 abstention, F16). No model, no index: tests the pure
`assess_evidence` decision over synthetic top-1 scores, plus the two guarantees the WP4 contract
relies on: (1) the verdict is always attached; (2) `suppress_on_insufficient` controls whether an
insufficient verdict withholds passages while never mutating the ranking itself.

The operating point (min_top_score=0.26, signal=rerank_score) and its measured tradeoff are
recorded in EVAL_LOG/DECISIONS 2026-10-28; this test pins the decision LOGIC, not the corpus
numbers.
"""

from __future__ import annotations

from project_rag.api import assess_evidence
from project_rag.config import AbstentionConfig
from project_rag.schema import RetrievedResult, SourceLocation


def _result(score_key: str, score: float) -> RetrievedResult:
    return RetrievedResult(
        chunk_id="c",
        document_id="d",
        source_path="p",
        source_location=SourceLocation(),
        excerpt="x",
        rank=1,
        retrieval_channels=["dense", "lexical"],
        score_details={score_key: score},
    )


def test_disabled_gate_is_always_sufficient() -> None:
    cfg = AbstentionConfig(enabled=False)
    v, warns = assess_evidence([_result("rerank_score", 0.0)], cfg)
    assert v.enabled is False and v.sufficient is True and warns == []


def test_sufficient_above_threshold() -> None:
    cfg = AbstentionConfig(enabled=True, signal="rerank_score", min_top_score=0.26)
    v, _ = assess_evidence([_result("rerank_score", 0.8)], cfg)
    assert v.enabled and v.sufficient and v.top_score == 0.8 and v.threshold == 0.26


def test_insufficient_below_threshold() -> None:
    cfg = AbstentionConfig(enabled=True, signal="rerank_score", min_top_score=0.26)
    v, _ = assess_evidence([_result("rerank_score", 0.05)], cfg)
    assert v.enabled and v.sufficient is False and v.top_score == 0.05


def test_boundary_is_inclusive() -> None:
    # top == threshold counts as sufficient (>=).
    cfg = AbstentionConfig(enabled=True, min_top_score=0.26)
    v, _ = assess_evidence([_result("rerank_score", 0.26)], cfg)
    assert v.sufficient is True


def test_empty_results_are_insufficient_when_enabled() -> None:
    cfg = AbstentionConfig(enabled=True, min_top_score=0.26)
    v, _ = assess_evidence([], cfg)
    assert v.enabled and v.sufficient is False and v.top_score is None


def test_rerank_signal_missing_deactivates_gate_with_warning() -> None:
    # Gate asks for rerank_score but the reranker is off → only fusion_score present. The gate
    # must NOT silently treat a missing calibrated score as insufficient; it deactivates + warns
    # (v1 §4.6: raw fusion scores are not calibrated confidence).
    cfg = AbstentionConfig(enabled=True, signal="rerank_score", min_top_score=0.26)
    v, warns = assess_evidence([_result("fusion_score", 0.03)], cfg)
    assert v.enabled is False and v.sufficient is True
    assert any("rerank_score absent" in w for w in warns)
