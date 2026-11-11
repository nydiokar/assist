"""Owner-session log-row contract (Round-2 Task C). Pure, offline.

Pins the JSONL schema the owner-drivable eval (scripts/ask.py) writes, so the honest-eval record
stays stable and machine-readable: query, evidence verdict + top score, citations with source path
and rerank score, and the owner's own rating slot.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from project_rag.schema import (
    EvidenceVerdict,
    RetrievedResult,
    SearchResponse,
    SourceLocation,
)

_ASK = Path(__file__).resolve().parents[2] / "scripts" / "ask.py"
_spec = importlib.util.spec_from_file_location("ask_cli", _ASK)
ask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ask)


def _response() -> SearchResponse:
    return SearchResponse(
        query_id="q",
        project_id="ypq",
        index_generation="g",
        evidence=EvidenceVerdict(
            enabled=True, sufficient=True, signal="rerank_score", top_score=0.91, threshold=0.26
        ),
        results=[
            RetrievedResult(
                chunk_id="c1",
                document_id="d1",
                source_path="proposal.docx",
                source_location=SourceLocation(heading="Budget"),
                excerpt="total budget 49 697,00",
                rank=1,
                retrieval_channels=["dense", "lexical"],
                score_details={"rerank_score": 0.91},
            )
        ],
    )


def test_log_row_shape_and_rating():
    row = ask._log_row(_response(), "what is the budget?", 4, "spot on")
    assert row["query"] == "what is the budget?"
    assert row["evidence_sufficient"] is True
    assert row["evidence_top_score"] == 0.91
    assert row["n_results"] == 1
    assert row["citations"][0]["source_path"] == "proposal.docx"
    assert row["citations"][0]["rerank_score"] == 0.91
    assert row["owner_rating"] == 4
    assert row["owner_note"] == "spot on"
    assert "ts" in row


def test_log_row_skipped_rating_is_none():
    row = ask._log_row(_response(), "q", None, "")
    assert row["owner_rating"] is None


def test_log_row_abstained_response_has_no_citations():
    res = SearchResponse(
        query_id="q",
        project_id="ypq",
        index_generation="g",
        evidence=EvidenceVerdict(
            enabled=True, sufficient=False, signal="rerank_score", top_score=0.003, threshold=0.26
        ),
        results=[],
    )
    row = ask._log_row(res, "weather?", None, "")
    assert row["evidence_sufficient"] is False
    assert row["n_results"] == 0
    assert row["citations"] == []
