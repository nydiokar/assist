"""Diagnostic: what does retrieval pull for abstained report questions BEFORE the gate suppresses?

Temporarily disables suppression to see the real top-5 + scores, so we can tell a true miss from
a gate-too-strict case. No LLM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from project_rag.api import ProjectRAG
from project_rag.evaluation import load_questions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--project", default="ypq")
    ap.add_argument("--root", default="yspdb")
    ap.add_argument("--questions", default="eval/yp_report_questions_candidate.jsonl")
    ap.add_argument("--only", nargs="*", default=None, help="only these ids")
    args = ap.parse_args()

    rag = ProjectRAG(config_path=args.config)
    # turn OFF suppression so we always see the raw ranked passages + the substring presence
    rag.config.retrieval.abstention.suppress_on_insufficient = False
    rag.register_project_root(args.project, args.root)
    questions = load_questions(args.questions)
    if args.only:
        questions = [q for q in questions if q.id in args.only]
    try:
        for q in questions:
            res = rag.search_project(args.project, q.query, top_k=5,
                                     filters={"doc_class": "project_record"})
            v = res.evidence
            ts = f"{v.top_score:.3f}" if v.top_score is not None else "n/a"
            print(f"\n### {q.id} [{'SUFF' if v.sufficient else 'ABST'} top={ts}] expect={q.expect_source_paths}")
            print(f"    want substr: {q.expect_substrings}")
            for r in res.results:
                rr = r.score_details.get("rerank_score")
                rr_s = f"{rr:.3f}" if rr is not None else "?"
                has = all(s in (r.expanded_context or r.excerpt) for s in q.expect_substrings)
                mark = "  <== HAS EXPECTED SUBSTR" if has else ""
                print(f"    [{r.rank}] rr={rr_s} {Path(r.source_path).name[:48]}{mark}")
    finally:
        rag.close()


if __name__ == "__main__":
    main()
