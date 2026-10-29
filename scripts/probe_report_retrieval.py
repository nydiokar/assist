"""Diagnostic: what does retrieval surface for the Task D report questions? No LLM.

Shows per question: evidence verdict (sufficient/abstain + top score), whether the expected source
is in top-5, whether the expected substring is in the served excerpts, and the top-3 served paths.
Runs twice: unfiltered, and with doc_class=project_record (the Task D reference-bleed fix).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from project_rag.api import ProjectRAG
from project_rag.evaluation import load_questions


def run(rag: ProjectRAG, project: str, questions, doc_class: str | None) -> None:
    label = doc_class or "ALL"
    hits = ev_hits = suff = 0
    print(f"\n===== doc_class={label} =====")
    for q in questions:
        filters = {"doc_class": doc_class} if doc_class else None
        res = rag.search_project(project, q.query, top_k=5, filters=filters)
        paths = [r.source_path for r in res.results]
        hit = any(p in q.expect_source_paths for p in paths)
        served = " ".join(r.expanded_context or r.excerpt for r in res.results)
        ev_hit = all(s in served for s in q.expect_substrings) if res.results else False
        verdict = res.evidence
        hits += int(hit)
        ev_hits += int(ev_hit)
        suff += int(verdict.sufficient and verdict.enabled)
        ts = f"{verdict.top_score:.3f}" if verdict.top_score is not None else "n/a"
        state = "SUFF" if (verdict.enabled and verdict.sufficient) else ("ABST" if verdict.enabled else "off")
        top3 = [Path(p).name[:34] for p in paths[:3]]
        print(f"  {q.id} [{state} ts={ts}] "
              f"src@5={'Y' if hit else 'n'} ev@5={'Y' if ev_hit else 'n'}  top: {top3}")
    n = len(questions)
    print(f"  -- doc_class={label}: src@5 {hits}/{n}={hits/n:.2f}  "
          f"ev@5 {ev_hits}/{n}={ev_hits/n:.2f}  sufficient {suff}/{n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--project", default="ypq")
    ap.add_argument("--root", default="yspdb")
    ap.add_argument("--questions", default="eval/yp_report_questions_candidate.jsonl")
    args = ap.parse_args()
    questions = load_questions(args.questions)
    rag = ProjectRAG(config_path=args.config)
    rag.register_project_root(args.project, args.root)
    try:
        run(rag, args.project, questions, None)
        run(rag, args.project, questions, "project_record")
    finally:
        rag.close()


if __name__ == "__main__":
    main()
