"""Evaluation harness runs over the synthetic corpus and meets a baseline recall bar.

This is a fixture-only gate (HashEmbedder). It proves the harness + metrics work end-to-end,
NOT that the production E5 system meets any quality SLA (v1 §6.2: no premature SLA).
"""

from __future__ import annotations

from pathlib import Path

from project_rag.evaluation import evaluate, load_questions

QUESTIONS = Path(__file__).resolve().parents[2] / "eval" / "archive" / "questions_synthetic_gv.jsonl"


def test_eval_harness_runs_and_meets_baseline(make_rag, corpus_root):
    rag = make_rag()
    rag.ingest_project("demo", corpus_root)
    questions = load_questions(QUESTIONS)
    report = evaluate(rag, "demo", questions, top_k=5)
    assert report.n == len(questions)
    # Lexical+hash fusion should locate an expected source for the clear factual questions.
    assert report.recall_at_5 >= 0.7, report.model_dump()
    assert report.mrr > 0.0
