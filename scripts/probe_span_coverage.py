"""Task A diagnostic: answer-span coverage under parent-section expansion (no reindex).

For each scored question, runs the production reranked pipeline, then for every top-5 result
that lands in an expected source, re-reads the source and extracts the "parent section" the
chunk came from (blocks sharing the chunk's heading; page for PDFs; sheet for tables). Measures:

  - evidence@5  : expected substring already in a top-5 CHUNK excerpt (current metric)
  - span@5      : expected substring present in the EXPANDED context of a top-5 doc hit

span@5 is the honest ceiling parent-section expansion can reach: it is the fraction of questions
for which, given the files we already retrieve at @5, the answer text is recoverable by handing
the LLM the surrounding section instead of only the top chunk. No LLM. WSL ROCm GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_rag.api import ProjectRAG
from project_rag.evaluation import load_questions
from project_rag.expand import expand_result_context


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--project", default="ypq")
    ap.add_argument("--root", default="yspdb")
    ap.add_argument("--questions", default="eval/yp_questions_v2.jsonl")
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--out", default="eval/results/span_coverage.json")
    args = ap.parse_args()

    qs = load_questions(args.questions)
    rag = ProjectRAG(config_path=args.config)
    rag.register_project_root(args.project, args.root)

    ev_hits = span_hits = scored = 0
    recovered: list[str] = []
    rows: list[dict] = []
    try:
        for q in qs:
            if q.kind in {"no_answer", "filesystem"} or not q.expect_source_paths:
                continue
            scored += 1
            res = rag.search_project(args.project, q.query, top_k=5)
            from_expected = [r for r in res.results if r.source_path in q.expect_source_paths]
            # current evidence@5: substring in a top-5 chunk excerpt of an expected source
            ev = bool(from_expected) and all(
                any(sub in r.excerpt for r in from_expected) for sub in q.expect_substrings
            )
            # span@5: substring in the EXPANDED context of any top-5 expected-source hit
            expanded_texts: list[str] = []
            for r in from_expected:
                ctx = expand_result_context(rag, args.project, r, max_chars=args.max_chars)
                expanded_texts.append(ctx)
            span = bool(expanded_texts) and all(
                any(sub in ex for ex in expanded_texts) for sub in q.expect_substrings
            )
            ev_hits += int(ev)
            span_hits += int(span)
            if span and not ev:
                recovered.append(q.id)
            rows.append(
                {
                    "id": q.id,
                    "kind": q.kind,
                    "doc_hit": bool(from_expected),
                    "evidence5": ev,
                    "span5": span,
                }
            )
    finally:
        rag.close()

    print(f"scored={scored}")
    print(f"evidence@5 = {ev_hits/scored:.4f} ({ev_hits}/{scored})")
    print(f"span@5     = {span_hits/scored:.4f} ({span_hits}/{scored})")
    print(f"recovered by expansion (span hit, evidence miss): {recovered}")
    for r in rows:
        if not r["evidence5"] or not r["span5"]:
            print(
                f"  {r['id']:6} {r['kind']:14} doc_hit={r['doc_hit']!s:5} "
                f"ev5={r['evidence5']!s:5} span5={r['span5']!s:5}"
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "scored": scored,
                "evidence_at_5": ev_hits / scored,
                "span_at_5": span_hits / scored,
                "recovered": recovered,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
