"""End-to-end WP4 proof: a real Claude agent answers the eval questions grounded ONLY in the RAG
MCP tools, and abstains on the known-absent ones.

This closes the loop the smoke test could not: the smoke test proves the stdio MCP server serves
retrieval; THIS proves the model reasons over that retrieval, cites it, and refuses to fabricate.

How it runs (no backend/harness change, no API key — the agent runs on the owner's plan):
  - `claude -p` is launched once per question IN THE PROJECT DIR, so Claude Code loads the RAG
    server from `.mcp.json` on its own (the normal project-scoped MCP path).
  - `--strict-mcp-config --mcp-config .mcp.json` → ONLY the RAG server is available.
  - `--allowed-tools mcp__project-rag__search_project ... open_source ... get_ingestion_status`
    → the agent can ONLY use RAG tools; it cannot read raw corpus files, so any grounded answer
    came through retrieval (real grounding, not a filesystem peek).
  - The grounding contract (answerer.SYSTEM_PROMPT) is appended as the system prompt so abstention
    behaviour travels into the session.

Checks per question:
  - answerable  -> NOT the abstention line, and every expected ground-truth substring present.
  - no_answer   -> the exact abstention line (model refused to fabricate a plausible-but-absent fact).

Run in WSL (authed `claude`, GPU index, .mcp.json present):
  cd /mnt/c/Users/Cicada38/Projects/assist
  PYTHONPATH=src /home/gorila/venvs/tokens-ingest-rocm/bin/python -X utf8 \
      scripts/eval_agent.py --model claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from project_rag.interfaces.mcp.tools import RAG_SERVER_NAME, RAG_TOOL_NAMES
from project_rag.services.answerer import ABSTENTION_LINE, SYSTEM_PROMPT

# Execution directive layered ON TOP of the pure grounding contract (SYSTEM_PROMPT). It makes the
# agent actually CALL the tool instead of answering from a first glance, and read the exact value
# (r03 earlier returned a per-month sub-figure, 4500, instead of the total 49 697). It does NOT
# relax grounding/abstention — it enforces the retrieval step the contract assumes.
EXECUTION_DIRECTIVE = (
    "\n\nEXECUTION — read before answering:\n"
    "A. You have ONE evidence tool, search_project, and NO file access. For EVERY question you "
    "MUST call search_project first (rephrase and call again if needed) before answering.\n"
    "B. NEVER do arithmetic. Do NOT add, multiply, or combine numbers from passages to produce a "
    "total/subtotal. Only report a figure that a passage states VERBATIM as the answer to the "
    "exact thing asked. If the question asks for a 'total X' and no passage literally labels a "
    f'number as the total X, you do NOT have it — reply exactly "{ABSTENTION_LINE}".\n'
    "C. A passage about the same TOPIC is not the same as a passage that states the asked FACT. "
    "If the asked fact (e.g. a specific 'catering budget', a 'participation fee', a 'hotel name') "
    "is not written explicitly in a passage, the fact is NOT established — do NOT substitute a "
    f'related number. Reply exactly "{ABSTENTION_LINE}" and nothing else.'
)
AGENT_SYSTEM_PROMPT = SYSTEM_PROMPT + EXECUTION_DIRECTIVE

QUESTIONS_PATH = Path("eval/yp_report_questions.jsonl")
RESULTS_PATH = Path("eval/results/agent_eval.jsonl")


def _allowed_tools() -> list[str]:
    # FQN form Claude Code uses for MCP tools: mcp__<server>__<tool>
    return [f"mcp__{RAG_SERVER_NAME}__{name}" for name in RAG_TOOL_NAMES]


def _ask_agent(query: str, model: str, mcp_config: str, timeout_s: int) -> tuple[str, dict]:
    """Launch one `claude -p` turn restricted to the RAG tools. Returns (answer_text, raw_json)."""
    cmd = [
        "claude",
        "-p",
        query,
        "--model",
        model,
        "--output-format",
        "json",
        "--mcp-config",
        mcp_config,
        "--strict-mcp-config",
        "--permission-mode",
        "bypassPermissions",
        "--allowed-tools",
        *_allowed_tools(),
        "--append-system-prompt",
        AGENT_SYSTEM_PROMPT,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude exited {proc.returncode}: {proc.stderr[:500] or proc.stdout[:500]}"
        )
    raw = json.loads(proc.stdout)
    # json output format: {"type":"result","result": "<assistant text>","num_turns": N, ...}
    # It carries NO transcript (`messages`) — tool use is inferred from num_turns (see _used_rag).
    answer = raw.get("result") if isinstance(raw, dict) else str(raw)
    return (answer or "").strip(), raw


def _norm(text: str) -> str:
    """Casefold and fold en/em dashes to a plain hyphen, so a correct answer is not failed on
    casing or a typographic dash variant in a date range / name."""
    return text.casefold().replace("–", "-").replace("—", "-").replace("−", "-")


def _abstained(answer: str) -> bool:
    # Model may wrap/translate; treat a leading/standalone match of the fixed line as abstention.
    norm = answer.strip().strip('"').rstrip(".")
    return norm == ABSTENTION_LINE.rstrip(".") or ABSTENTION_LINE in answer


def _used_rag(raw: dict) -> bool:
    """True if the agent made at least one tool round-trip. The `json` output format carries no
    transcript, so we use `num_turns`: a pure text answer is 1 turn; each tool call + its result
    adds turns. With tools as the ONLY capability (strict + allowed-tools = RAG only), num_turns>1
    means the agent called a RAG tool before answering. permission_denials would flag a blocked
    attempt; we also treat any denial as 'attempted use'."""
    if not isinstance(raw, dict):
        return False
    if raw.get("permission_denials"):
        return True
    return int(raw.get("num_turns", 1)) > 1


def _check(item: dict, answer: str) -> tuple[bool, str]:
    is_no_answer = item.get("kind") == "no_answer" or not item.get("expect_substrings")
    if is_no_answer:
        if _abstained(answer):
            return True, "abstained as expected"
        return False, "FABRICATED — expected abstention on absent fact"
    if _abstained(answer):
        return False, "abstained on an ANSWERABLE question"
    # Normalise before matching: ground-truth labels are verbatim, but a correct answer may re-case
    # (org names) or substitute a typographically-equivalent dash (e.g. a date range "17–22" with an
    # en-dash vs the "-" in the label). Neither changes the fact. Numeric digits match regardless.
    missing = [s for s in item.get("expect_substrings", []) if _norm(s) not in _norm(answer)]
    if missing:
        return False, f"missing ground-truth substrings: {missing}"
    return True, "grounded with all expected facts"


def main() -> None:
    ap = argparse.ArgumentParser()
    # Default to sonnet: on the budget-total question (r03) Haiku mis-read which labeled line was
    # the answer (reported the first 'Project Costs' sub-line, not the line literally labeled
    # 'Total'); sonnet reads the same retrieved passage correctly. Retrieval/wiring are identical —
    # the gap is model reading capability (ledger F21). Haiku remains runnable via --model.
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--mcp-config", default=".mcp.json")
    ap.add_argument("--questions", default=str(QUESTIONS_PATH))
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--only", default=None, help="comma-separated ids to run a subset")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    items = [
        json.loads(ln)
        for ln in Path(args.questions).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if only:
        items = [it for it in items if it["id"] in only]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = RESULTS_PATH.open("w", encoding="utf-8")

    npass = 0
    for it in items:
        qid, query = it["id"], it["query"]
        print(f"\n=== {qid} [{it.get('kind','?')}] {query[:70]}")
        try:
            answer, raw = _ask_agent(query, args.model, args.mcp_config, args.timeout)
        except Exception as exc:  # noqa: BLE001 - record failure, continue the sweep
            print(f"  ERROR: {exc}")
            out.write(json.dumps({"id": qid, "error": str(exc)}, ensure_ascii=False) + "\n")
            continue
        ok, reason = _check(it, answer)
        used = _used_rag(raw)
        npass += int(ok)
        print(f"  tool_used={used}  ok={ok}  {reason}")
        print(f"  ANSWER: {answer[:280]}")
        out.write(
            json.dumps(
                {
                    "id": qid,
                    "kind": it.get("kind"),
                    "query": query,
                    "answer": answer,
                    "ok": ok,
                    "reason": reason,
                    "rag_tool_used": used,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    out.close()
    total = len(items)
    print(f"\n=== AGENT EVAL: {npass}/{total} passed ===  results -> {RESULTS_PATH}")
    raise SystemExit(0 if npass == total else 1)


if __name__ == "__main__":
    main()
