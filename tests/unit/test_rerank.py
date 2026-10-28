"""Cross-encoder reranker reorder logic (F19). No model download: a fake scorer is injected
into the lazy `_model` slot so `_ensure` is never called. Verifies the three guarantees the
pipeline relies on: (1) the top_n head is reordered by descending score, (2) candidates beyond
top_n keep their fused order after the reranked head (pool recall never shrinks), (3) ranks are
renumbered 1..n and rerank_score is recorded for audit.
"""

from __future__ import annotations

from project_rag.config import RerankConfig
from project_rag.rerank import CrossEncoderReranker
from project_rag.schema import RetrievedResult, SourceLocation


class _FakeCrossEncoder:
    """Returns a preset score per (query, excerpt). Deterministic, no network."""

    def __init__(self, scores_by_excerpt: dict[str, float]) -> None:
        self._scores = scores_by_excerpt

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]:
        return [self._scores[excerpt] for _query, excerpt in pairs]


def _result(chunk_id: str, excerpt: str, rank: int) -> RetrievedResult:
    return RetrievedResult(
        chunk_id=chunk_id, document_id="d", source_path="p",
        source_location=SourceLocation(), excerpt=excerpt, rank=rank,
        retrieval_channels=["dense", "lexical"], score_details={"fusion_score": 1.0 / rank},
    )


def _reranker(scores: dict[str, float], top_n: int) -> CrossEncoderReranker:
    rr = CrossEncoderReranker(RerankConfig(enabled=True, top_n=top_n))
    rr._model = _FakeCrossEncoder(scores)  # inject fake, skip _ensure/download
    return rr


def test_rerank_reorders_head_by_score() -> None:
    # Fused order A,B,C; reranker thinks C is most relevant, then A, then B.
    results = [_result("A", "exA", 1), _result("B", "exB", 2), _result("C", "exC", 3)]
    out = _reranker({"exA": 0.5, "exB": 0.1, "exC": 0.9}, top_n=3).rerank("q", results)
    assert [r.chunk_id for r in out] == ["C", "A", "B"]
    assert [r.rank for r in out] == [1, 2, 3]
    assert out[0].score_details["rerank_score"] == 0.9


def test_rerank_preserves_tail_after_window() -> None:
    # top_n=2: only A,B are rerankable; C,D keep their fused order behind the reranked head.
    results = [_result(c, f"ex{c}", i) for i, c in enumerate("ABCD", start=1)]
    out = _reranker({"exA": 0.1, "exB": 0.9, "exC": 0.0, "exD": 0.0}, top_n=2).rerank("q", results)
    assert [r.chunk_id for r in out] == ["B", "A", "C", "D"]
    assert [r.rank for r in out] == [1, 2, 3, 4]


def test_rerank_tie_break_is_deterministic() -> None:
    # Equal scores → stable tie-break on chunk_id, not input order.
    results = [_result("Zed", "exZ", 1), _result("Abe", "exA", 2)]
    out = _reranker({"exZ": 0.5, "exA": 0.5}, top_n=2).rerank("q", results)
    assert [r.chunk_id for r in out] == ["Abe", "Zed"]


def test_rerank_empty_is_noop() -> None:
    assert _reranker({}, top_n=12).rerank("q", []) == []
