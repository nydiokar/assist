"""Grounded project-assistant answerer (WP4, Option 2b — owner egress decision 2026-10-29).

The demo surface the owner asked for: he types his own questions and gets a grounded, cited answer
drawn ONLY from the retrieved corpus passages, with honest abstention when the evidence gate finds
nothing. Retrieval-first (v1 §2.2): `search_project` runs and gates before any LLM call; on an
abstain verdict the answer is the fixed "Not established…" line with NO token spent.

EGRESS: retrieved Bulgarian passages go to Anthropic (hosted Claude). Authorized by the written
owner decision in docs/DECISIONS.md (2026-10-29) per v1 §8. Set ANTHROPIC_API_KEY in the
environment; the key is never read from a file or hardcoded.

This is the LLM-wired sibling of scripts/ask.py (which is retrieval-only). Same warm-retriever
session shape: one ProjectRAG for the whole run (cold start ~30 s paid once).

Run (WSL ROCm GPU for the retriever; -X utf8 for Bulgarian):
  wsl.exe -d Ubuntu-24.04 -- bash -c 'export HSA_ENABLE_DXG_DETECTION=1; export ANTHROPIC_API_KEY=...; \
    cd /mnt/c/Users/Cicada38/Projects/assist && PYTHONPATH=src \
    /home/gorila/venvs/tokens-ingest-rocm/bin/python -X utf8 scripts/answer.py \
    --config configs/yp_qwen4b_gpu_rerank.yaml --project ypq --root yspdb'

Batch replay (one question per line, no interactive rating):  add --questions FILE.
Scope to project material only (stop third-party reference-material bleed, the Task D finding):
  --doc-class project_record
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

from project_rag.answerer import (
    ABSTENTION_LINE,
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    answer_from_search,
    build_grounding_prompt,
    stream_claude,
)
from project_rag.api import ProjectRAG


def _log_row(question: str, ans, rating: int | None, note: str) -> dict:
    return {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": ans.answer,
        "abstained_pre_llm": ans.abstained_pre_llm,
        "evidence_sufficient": ans.evidence_sufficient,
        "evidence_top_score": ans.evidence_top_score,
        "model": ans.model,
        "citations": ans.citations,
        "owner_rating": rating,       # 1-5, owner's own judgement of usefulness (None if skipped)
        "owner_note": note,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--project", default="ypq")
    ap.add_argument("--root", default="yspdb")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="hosted Claude model (owner egress decision 2026-10-29)")
    ap.add_argument("--doc-class", default=None,
                    choices=["project_record", "reference", "filesystem"],
                    help="scope retrieval to one material class "
                         "(use project_record for report-drafting; Task D finding)")
    ap.add_argument("--questions", default=None,
                    help="optional file of questions (one per line) for non-interactive replay")
    ap.add_argument("--session-log", default=None,
                    help="JSONL to append to (default eval/results/answer_session_<date>.jsonl)")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. This answerer sends retrieved corpus passages "
              "to Anthropic (hosted Claude) — authorized by docs/DECISIONS.md 2026-10-29. "
              "Export the key and re-run; it is never read from a file.", file=sys.stderr)
        sys.exit(2)

    log_path = Path(args.session_log or
                    f"eval/results/answer_session_{dt.date.today().isoformat()}.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading retriever (warm for the whole session; cold start ~30 s)…")
    rag = ProjectRAG(config_path=args.config)
    rag.register_project_root(args.project, args.root)
    filters = {"doc_class": args.doc_class} if args.doc_class else None

    batch: list[str] | None = None
    if args.questions:
        batch = [ln.strip() for ln in Path(args.questions).read_text(encoding="utf-8").splitlines()
                 if ln.strip()]

    print(f"Ready. Project={args.project}. Model={args.model}. "
          f"doc_class={args.doc_class or 'all'}. Session log: {log_path}")
    if batch is None:
        print("Type a question (BG or EN). Blank line or 'q' to quit.\n")

    try:
        idx = 0
        while True:
            if batch is not None:
                if idx >= len(batch):
                    break
                question = batch[idx]
                idx += 1
                print(f"\nQ> {question}")
            else:
                try:
                    question = input("Q> ").strip()
                except EOFError:
                    break
                if not question or question.lower() in {"q", "quit", "exit"}:
                    break

            res = rag.search_project(args.project, question, top_k=args.top_k, filters=filters)

            # Retrieval-first: if the gate suppressed passages, abstain with no LLM call / no egress.
            if not res.results:
                ans = answer_from_search(question, res, model=args.model)
                print(f"\n  [no LLM call — retrieval gate abstained]\n  {ans.answer}")
                ev = res.evidence
                if ev.enabled and ev.top_score is not None:
                    print(f"  (top relevance {ev.top_score:.3f} < threshold {ev.threshold})")
            else:
                # Stream the grounded answer token-by-token for the demo.
                print("\n  Answer:")
                user = build_grounding_prompt(question, res)
                acc: list[str] = []
                for piece in stream_claude(SYSTEM_PROMPT, user, args.model):
                    acc.append(piece)
                    print(piece, end="", flush=True)
                print()
                # Rebuild the structured record (reuses the pure path; no second LLM call).
                answer_text = "".join(acc).strip()
                ans = answer_from_search(question, res,
                                         call_claude=lambda *_: answer_text, model=args.model)
                print("\n  Cited passages:")
                for i, c in enumerate(ans.citations, start=1):
                    print(f"    [#{i}] {c}")

            rating: int | None = None
            note = ""
            if batch is None:
                raw = input("\n  your usefulness rating 1-5 (Enter to skip, 'q' to quit): ").strip()
                if raw.lower() in {"q", "quit", "exit"}:
                    _append(log_path, _log_row(question, ans, None, ""))
                    break
                if raw in {"1", "2", "3", "4", "5"}:
                    rating = int(raw)
                note = input("  note (optional, Enter to skip): ").strip()

            _append(log_path, _log_row(question, ans, rating, note))
            print("  — logged —\n")
    finally:
        rag.close()
    print(f"\nSession saved to {log_path}")


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
