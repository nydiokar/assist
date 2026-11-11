"""Golden snapshot of end-to-end behaviour on the synthetic fixture corpus (v3 build plan S0.3).

Pins what the A0 refactor (S1) must keep byte-identical: the ingest report, the generation id,
the status report, and for every fixture question the served chunk ids, their order, rounded
scores and the evidence verdict. Two configurations are captured: the shipped defaults, and a
variant that switches on the query-time paths (fusion-score abstention with suppression, and
parent-section context expansion) so those code paths are pinned too.

The test fails on ANY diff. A stage that changes behaviour on purpose regenerates the file with
`UPDATE_GOLDEN=1 uv run pytest tests/evaluation/test_golden_snapshot.py` and records the diff in
its build-plan log entry.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from project_rag.api import ProjectRAG
from project_rag.evaluation import load_questions
from tests.conftest import _config

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "eval" / "archive" / "questions_synthetic_gv.jsonl"
GOLDEN = Path(__file__).resolve().parent / "golden" / "v2_baseline.json"
SCORE_DECIMALS = 6

VARIANTS: dict[str, dict[str, object]] = {
    "defaults": {},
    "abstain_expand": {
        "retrieval.abstention.enabled": True,
        "retrieval.abstention.signal": "fusion_score",
        "retrieval.abstention.min_top_score": 0.03,
        "retrieval.abstention.suppress_on_insufficient": True,
        "retrieval.context_expansion.enabled": True,
    },
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, SCORE_DECIMALS)


def _capture(rag: ProjectRAG, corpus_root: str) -> dict[str, object]:
    report = rag.ingest_project("demo", corpus_root)
    again = rag.ingest_project("demo", corpus_root)
    status = rag.get_ingestion_status("demo")
    queries: dict[str, object] = {}
    for q in load_questions(QUESTIONS):
        resp = rag.search_project("demo", q.query, top_k=5)
        queries[q.id] = {
            "index_generation": resp.index_generation,
            "query_id": resp.query_id,
            "verdict": {
                "enabled": resp.evidence.enabled,
                "sufficient": resp.evidence.sufficient,
                "signal": resp.evidence.signal,
                "top_score": _round(resp.evidence.top_score),
                "threshold": resp.evidence.threshold,
            },
            "warnings": resp.warnings,
            "results": [
                {
                    "rank": r.rank,
                    "chunk_id": r.chunk_id,
                    "source_path": r.source_path,
                    "channels": r.retrieval_channels,
                    "scores": {k: _round(v) for k, v in sorted(r.score_details.items())},
                    "excerpt_sha": _sha(r.excerpt),
                    "expanded_sha": _sha(r.expanded_context) if r.expanded_context else "",
                }
                for r in resp.results
            ],
        }
    return {
        "ingest": {
            "index_generation": report.index_generation,
            "added": report.added,
            "modified": report.modified,
            "deleted": report.deleted,
            "unchanged": report.unchanged,
            "embedding_profile_key": report.embedding_profile_key,
            "warnings": report.warnings,
            "files": [
                {
                    "source_path": f.source_path,
                    "state": str(f.state),
                    "source_id": f.source_id,
                    "version_id": f.version_id,
                    "n_blocks": f.n_blocks,
                    "n_chunks": f.n_chunks,
                }
                for f in sorted(report.files, key=lambda f: f.source_path)
            ],
        },
        "reingest": {
            "index_generation": again.index_generation,
            "added": again.added,
            "modified": again.modified,
            "deleted": again.deleted,
            "unchanged": again.unchanged,
            "warnings": again.warnings,
        },
        "status": {
            "index_generation": status.index_generation,
            "n_documents": status.n_documents,
            "n_chunks": status.n_chunks,
            "states": dict(sorted(status.states.items())),
            "healthy": status.healthy,
            "warnings": status.warnings,
        },
        "queries": queries,
    }


def _serialise(snapshot: dict[str, object]) -> str:
    return json.dumps(snapshot, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def test_golden_snapshot(tmp_path: Path, corpus_root: str) -> None:
    snapshot: dict[str, object] = {}
    for name, overrides in VARIANTS.items():
        # A fresh directory per variant: no manifest, store or stage/embedding cache is shared.
        rag = ProjectRAG(config=_config(tmp_path / name, **overrides))
        try:
            snapshot[name] = _capture(rag, corpus_root)
        finally:
            rag.close()
    actual = _serialise(snapshot)
    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(actual, encoding="utf-8", newline="\n")
        pytest.skip("golden snapshot regenerated")
    expected = GOLDEN.read_text(encoding="utf-8")
    assert actual == expected, "golden snapshot drifted; see module docstring"
