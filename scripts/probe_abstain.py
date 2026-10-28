"""Diagnostic for Task 2 (abstention): dump per-question retrieval-score signals.

Runs the full reranked pipeline on a question set and prints, per question:
  kind, hit@5, top-1 rerank_score, top-2 rerank_score, margin (top1-top2), mean(top5).
Used to find a signal that separates no_answer (irrelevant-but-plausible) from answerable.
No LLM. WSL ROCm GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_rag.api import ProjectRAG
from project_rag.evaluation import load_questions


def _score(r, key: str) -> float | None:
    v = r.score_details.get(key)
    return float(v) if v is not None else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--project", default="ypq")
    ap.add_argument("--root", default="yspdb")
    ap.add_argument("--questions", default="eval/yp_questions_v2.jsonl")
    ap.add_argument("--signal", default="rerank_score",
                    help="score_details key to probe (rerank_score | fusion_score)")
    ap.add_argument("--out", default="eval/results/abstain_probe.json")
    args = ap.parse_args()

    qs = load_questions(args.questions)
    rag = ProjectRAG(config_path=args.config)
    rag.register_project_root(args.project, args.root)
    rows: list[dict] = []
    try:
        for q in qs:
            res = rag.search_project(args.project, q.query, top_k=5)
            scores = [_score(r, args.signal) for r in res.results]
            scores = [s for s in scores if s is not None]
            top1 = scores[0] if scores else None
            top2 = scores[1] if len(scores) > 1 else None
            margin = (top1 - top2) if (top1 is not None and top2 is not None) else None
            mean5 = (sum(scores) / len(scores)) if scores else None
            top_paths = [r.source_path for r in res.results]
            hit = (q.kind != "no_answer"
                   and any(p in q.expect_source_paths for p in top_paths))
            rows.append({
                "id": q.id, "kind": q.kind, "no_answer": q.kind == "no_answer",
                "hit5": hit, "top1": top1, "top2": top2, "margin": margin, "mean5": mean5,
            })
    finally:
        rag.close()

    rows.sort(key=lambda r: (r["top1"] if r["top1"] is not None else -1e9))
    print(f"signal={args.signal}  (sorted ascending by top1)")
    print(f"{'id':6} {'kind':14} {'noAns':5} {'hit5':4} {'top1':>9} {'top2':>9} {'margin':>8} {'mean5':>9}")
    for r in rows:
        def f(x): return f"{x:9.3f}" if isinstance(x, float) else f"{'--':>9}"
        print(f"{r['id']:6} {r['kind']:14} {str(r['no_answer']):5} {str(r['hit5']):4} "
              f"{f(r['top1'])} {f(r['top2'])} "
              f"{(f'{r['margin']:8.3f}' if r['margin'] is not None else '      --')} {f(r['mean5'])}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
