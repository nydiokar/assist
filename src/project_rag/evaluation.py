"""Retrieval evaluation harness (v1 §6): Recall@k, MRR over labeled questions.

No LLM. Reads eval/questions.jsonl, runs search_project per question, computes whether an
expected source appears in the top-k. Writes a report artifact. `no_answer` questions are
scored as a separate abstention signal (expected empty / weak retrieval), not as recall.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .api import ProjectRAG


class EvalQuestion(BaseModel):
    id: str
    lang: str
    kind: str
    query: str
    expect_source_paths: list[str]
    expect_substrings: list[str]
    filters: dict = {}               # forwarded to search_project (doc_class, ...)


class QuestionResult(BaseModel):
    id: str
    kind: str
    hit_at_5: bool
    evidence_hit_at_5: bool = False   # expected substrings found in top-k excerpts of expected sources
    reciprocal_rank: float
    matched_substring: bool
    top_paths: list[str]


class EvalReport(BaseModel):
    n: int
    recall_at_5: float
    evidence_recall_at_5: float = 0.0
    mrr: float
    substring_rate: float
    per_question: list[QuestionResult]
    notes: list[str] = []


def load_questions(path: str | Path) -> list[EvalQuestion]:
    out: list[EvalQuestion] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(EvalQuestion.model_validate_json(line))
    return out


def evaluate(rag: ProjectRAG, project_id: str, questions: list[EvalQuestion],
             top_k: int = 5, channels: list[str] | None = None) -> EvalReport:
    per: list[QuestionResult] = []
    recall_hits = 0
    rr_sum = 0.0
    substr_hits = 0
    evidence_hits = 0
    scored = 0
    for q in questions:
        filters = {**q.filters, **({"channels": channels} if channels else {})}
        res = rag.search_project(project_id, q.query, top_k=top_k, filters=filters or None)
        top_paths = [r.source_path for r in res.results]
        joined = " ".join(r.excerpt for r in res.results)
        substr = all(s in joined for s in q.expect_substrings) if q.expect_substrings else True

        if q.kind == "no_answer" or not q.expect_source_paths:
            per.append(QuestionResult(id=q.id, kind=q.kind, hit_at_5=False,
                                      reciprocal_rank=0.0, matched_substring=substr,
                                      top_paths=top_paths))
            continue

        scored += 1
        from_expected = [r.excerpt for r in res.results if r.source_path in q.expect_source_paths]
        evidence = bool(from_expected) and all(
            any(sub in ex for ex in from_expected) for sub in q.expect_substrings)
        evidence_hits += int(evidence)
        rank = next((i for i, p in enumerate(top_paths, start=1)
                     if p in q.expect_source_paths), None)
        hit = rank is not None
        recall_hits += int(hit)
        rr_sum += (1.0 / rank) if rank else 0.0
        substr_hits += int(substr)
        per.append(QuestionResult(id=q.id, kind=q.kind, hit_at_5=hit, evidence_hit_at_5=evidence,
                                  reciprocal_rank=(1.0 / rank) if rank else 0.0,
                                  matched_substring=substr, top_paths=top_paths))

    return EvalReport(
        n=len(questions),
        recall_at_5=round(recall_hits / scored, 4) if scored else 0.0,
        evidence_recall_at_5=round(evidence_hits / scored, 4) if scored else 0.0,
        mrr=round(rr_sum / scored, 4) if scored else 0.0,
        substring_rate=round(substr_hits / scored, 4) if scored else 0.0,
        per_question=per,
        notes=[f"scored={scored} (excludes no_answer)",
               "embedder/store are configurable; results are fixture-only until real corpus"],
    )
