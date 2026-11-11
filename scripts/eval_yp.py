"""Real-corpus retrieval evaluation (spec v1 §6.1-6.2). No LLM.

1. Label validation: every expected substring must occur in the parsed text of an expected
   source (labels are checked against the corpus, not assumed).
2. Runs each channel set against the ingested index and prints Recall@5 / evidence recall / MRR.

    $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe -X utf8 scripts\\eval_yp.py --config configs\\yp_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_rag.api import ProjectRAG
from project_rag.config import load_config
from project_rag.discovery import _long_path
from project_rag.evaluation import EvalQuestion, evaluate, load_questions
from project_rag.parsers import ParserRegistry


def validate_labels(questions: list[EvalQuestion], root: Path) -> list[str]:
    reg = ParserRegistry(load_config(None).parsing)
    cache: dict[str, str] = {}
    problems: list[str] = []
    for q in questions:
        # filesystem-presence questions reference a FOLDER (synthetic doc_class=filesystem chunk),
        # not a parseable file — their labels are validated by construction, not by parsing.
        if q.kind == "filesystem":
            continue
        for sp in q.expect_source_paths:
            if sp not in cache:
                full = _long_path(str((root / sp).resolve()))
                doc = reg.parse(full, sp, Path(sp).suffix.lower())
                cache[sp] = "\n".join(b.text for b in doc.blocks)
        for sub in q.expect_substrings:
            if not any(sub in cache[sp] for sp in q.expect_source_paths):
                problems.append(f"{q.id}: substring not found in expected sources: {sub!r}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--project", default="yp")
    ap.add_argument("--root", default="yspdb")
    ap.add_argument("--questions", default="eval/yp_questions.jsonl")
    ap.add_argument(
        "--channels",
        nargs="*",
        default=["lexical"],
        help="channel sets to run, e.g. lexical dense lexical+dense",
    )
    ap.add_argument(
        "--top-k", type=int, default=5, help="k for Recall@k / evidence@k / MRR (default 5)"
    )
    ap.add_argument("--out", default="eval/results/yp_eval.json")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    problems = validate_labels(questions, Path(args.root))
    for p in problems:
        print("LABEL PROBLEM:", p)
    if problems:
        raise SystemExit(1)
    print(f"labels validated: {len(questions)} questions")

    rag = ProjectRAG(config_path=args.config)
    rag.register_project_root(args.project, args.root)
    results: dict[str, object] = {}
    try:
        for cs in args.channels:
            rep = evaluate(rag, args.project, questions, top_k=5, channels=cs.split("+"))
            results[cs] = rep.model_dump()
            print(
                f"\n== {cs}: Recall@5={rep.recall_at_5} evidence@5={rep.evidence_recall_at_5} "
                f"MRR={rep.mrr} (n={rep.n})"
            )
            if rep.abstention_enabled:
                print(
                    f"   abstention: no_answer withheld {rep.noanswer_abstained}/{rep.noanswer_total}; "
                    f"answerable wrongly withheld {rep.answerable_abstained}/{rep.answerable_total} "
                    f"(of which real hits: {rep.answerable_hit_abstained})"
                )
            for r in rep.per_question:
                mark = "HIT " if r.hit_at_5 else "miss"
                ev = "EV" if r.evidence_hit_at_5 else "--"
                ab = " ABSTAIN" if r.abstained else ""
                ts = f" ts={r.top_score:.3f}" if r.top_score is not None else ""
                print(f"  {r.id} {r.kind:15} {mark} {ev} rr={r.reciprocal_rank:.2f}{ts}{ab}")
    finally:
        rag.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
