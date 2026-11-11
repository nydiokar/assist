"""Owner-drivable retrieval session (Round-2 Task C). The honest eval when there is no gold set.

The automated eval set is self-authored — its credibility, not its score, is the bottleneck. This
lets the OWNER (who owns the corpus and knows the true answers) type his own questions and judge
usefulness himself. Per question it logs: the retrieved passages + citable source paths, the
evidence verdict (sufficient / abstained + top relevance score), the parent-section context handed
to a future LLM, and the owner's own 1–5 usefulness rating + note. The appended JSONL session IS
the trustworthy evaluation record, and this is also the backbone of the demo surface.

NO LLM — this is the retrieval layer the owner interrogates directly (retrieval-first, v1 §2.2).
The retriever is kept WARM (one ProjectRAG for the whole session; cold start ~30 s is paid once).

Run (WSL ROCm GPU, Bulgarian-safe with -X utf8):
  wsl.exe -d Ubuntu-24.04 -- bash -c 'export HSA_ENABLE_DXG_DETECTION=1; \
    cd /mnt/c/Users/Cicada38/Projects/assist && PYTHONPATH=src \
    /home/gorila/venvs/tokens-ingest-rocm/bin/python -X utf8 scripts/ask.py \
    --config configs/yp_qwen4b_gpu_rerank.yaml --project ypq --root yspdb'

Non-interactive batch (e.g. replay a file of questions, no ratings): add --questions FILE.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from project_rag.api import ProjectRAG
from project_rag.schema import SearchResponse


def _print_response(res: SearchResponse, show_context: bool) -> None:
    ev = res.evidence
    verdict = "SUFFICIENT" if ev.sufficient else "INSUFFICIENT — abstain"
    score = f"{ev.top_score:.3f}" if ev.top_score is not None else "n/a"
    print(
        f"\n  evidence: {verdict}  (signal={ev.signal or 'off'} top_score={score} "
        f"threshold={ev.threshold})"
    )
    if not res.results:
        print(
            "  → No sufficient evidence in the accessible project documents. "
            "(An assistant should decline rather than answer.)"
        )
    for r in res.results:
        rr = r.score_details.get("rerank_score")
        rr_s = f" rerank={rr:.3f}" if rr is not None else ""
        loc = r.source_location
        where = []
        if loc.page is not None:
            where.append(f"p.{loc.page}")
        if loc.sheet:
            where.append(f"sheet={loc.sheet}")
        if loc.heading:
            where.append(f"§{loc.heading[:40]}")
        where_s = (" [" + ", ".join(where) + "]") if where else ""
        print(f"\n  [{r.rank}] {r.source_path}{where_s}{rr_s}")
        print(f"      {r.excerpt[:300].strip().replace(chr(10), ' ')}")
        if show_context and r.expanded_context and r.expanded_context != r.excerpt:
            ctx = r.expanded_context[:500].strip().replace(chr(10), " ")
            print(f"      ⤷ section context: {ctx}")
    if res.warnings:
        print(f"\n  warnings: {res.warnings}")


def _log_row(res: SearchResponse, query: str, rating: int | None, note: str) -> dict:
    return {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "evidence_sufficient": res.evidence.sufficient,
        "evidence_signal": res.evidence.signal,
        "evidence_top_score": res.evidence.top_score,
        "n_results": len(res.results),
        "citations": [
            {
                "rank": r.rank,
                "source_path": r.source_path,
                "rerank_score": r.score_details.get("rerank_score"),
                "excerpt": r.excerpt[:500],
            }
            for r in res.results
        ],
        "owner_rating": rating,  # 1-5 usefulness, owner's own judgement (None if skipped)
        "owner_note": note,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--project", default="ypq")
    ap.add_argument("--root", default="yspdb")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--questions",
        default=None,
        help="optional file of questions (one per line) for non-interactive replay",
    )
    ap.add_argument(
        "--show-context",
        action="store_true",
        default=True,
        help="print the parent-section context handed to a future LLM",
    )
    ap.add_argument(
        "--session-log",
        default=None,
        help="JSONL to append to (default eval/results/owner_session_<date>.jsonl)",
    )
    args = ap.parse_args()

    log_path = Path(
        args.session_log or f"eval/results/owner_session_{dt.date.today().isoformat()}.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading retriever (warm for the whole session; cold start ~30 s)…")
    rag = ProjectRAG(config_path=args.config)
    rag.register_project_root(args.project, args.root)

    batch: list[str] | None = None
    if args.questions:
        batch = [
            ln.strip()
            for ln in Path(args.questions).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    print(f"Ready. Project={args.project}. Session log: {log_path}")
    if batch is None:
        print("Type a question (BG or EN). Blank line or 'q' to quit.\n")

    try:
        idx = 0
        while True:
            if batch is not None:
                if idx >= len(batch):
                    break
                query = batch[idx]
                idx += 1
                print(f"\nQ> {query}")
            else:
                try:
                    query = input("Q> ").strip()
                except EOFError:
                    break
                if not query or query.lower() in {"q", "quit", "exit"}:
                    break

            res = rag.search_project(args.project, query, top_k=args.top_k)
            _print_response(res, args.show_context)

            rating: int | None = None
            note = ""
            if batch is None:
                raw = input("\n  your usefulness rating 1-5 (Enter to skip, 'q' to quit): ").strip()
                if raw.lower() in {"q", "quit", "exit"}:
                    _append(log_path, _log_row(res, query, None, ""))
                    break
                if raw in {"1", "2", "3", "4", "5"}:
                    rating = int(raw)
                note = input("  note (optional, Enter to skip): ").strip()

            _append(log_path, _log_row(res, query, rating, note))
            print("  — logged —\n")
    finally:
        rag.close()
    print(f"\nSession saved to {log_path}")


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
