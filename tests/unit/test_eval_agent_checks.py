"""Offline unit tests for the agent-eval scorer (scripts/eval_agent.py). No claude, no GPU, no net.

Guards the three judgement rules that decide pass/fail of the live WP4 agent run:
  - abstention detection (exact fixed line, tolerant of quoting/trailing dot),
  - case-insensitive ground-truth substring match (a correct org name may be re-cased),
  - tool-use inference from the json result shape (num_turns / permission_denials).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "eval_agent", Path(__file__).resolve().parents[2] / "scripts" / "eval_agent.py"
)
assert _spec and _spec.loader
eval_agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_agent)

ABSTENTION = eval_agent.ABSTENTION_LINE


def test_abstained_detects_exact_and_quoted():
    assert eval_agent._abstained(ABSTENTION)
    assert eval_agent._abstained(f'"{ABSTENTION}"')
    assert eval_agent._abstained(ABSTENTION + " Документите не посочват конкретна сума.")
    assert not eval_agent._abstained("The total grant is 49 697,00 EUR.")


def test_check_no_answer_passes_on_abstention_fails_on_fabrication():
    item = {"id": "r11", "kind": "no_answer", "expect_substrings": []}
    ok, _ = eval_agent._check(item, ABSTENTION)
    assert ok
    bad, reason = eval_agent._check(item, "The catering budget was 2000 BGN.")
    assert not bad and "FABRICATED" in reason


def test_check_answerable_is_case_insensitive():
    item = {
        "id": "r04",
        "kind": "report_section",
        "expect_substrings": ["NATIONAL YOUTH FORUM BULGARIA"],
    }
    ok, _ = eval_agent._check(item, "The partner is National Youth Forum Bulgaria (E10179758).")
    assert ok


def test_check_answerable_is_dash_insensitive():
    # label uses a plain hyphen; a correct answer with an en-dash date range must still match
    item = {"id": "r02", "kind": "report_section", "expect_substrings": ["17-22 юни 2025"]}
    ok, _ = eval_agent._check(item, "Проведе се в Ямбол, 17–22 юни 2025 г.")
    assert ok


def test_check_answerable_fails_on_wrong_number():
    item = {"id": "r03", "kind": "report_section", "expect_substrings": ["49 697,00"]}
    bad, reason = eval_agent._check(item, "The total grant is 4,500.00 EUR.")
    assert not bad and "missing" in reason


def test_check_answerable_fails_when_it_abstains():
    item = {"id": "r03", "kind": "report_section", "expect_substrings": ["49 697,00"]}
    bad, reason = eval_agent._check(item, ABSTENTION)
    assert not bad and "abstained on an ANSWERABLE" in reason


def test_used_rag_from_num_turns_and_denials():
    assert eval_agent._used_rag({"num_turns": 4})
    assert not eval_agent._used_rag({"num_turns": 1})
    assert eval_agent._used_rag({"num_turns": 1, "permission_denials": [{"tool": "x"}]})


def test_allowed_tools_are_fqn_rag_names():
    tools = eval_agent._allowed_tools()
    assert tools == [
        "mcp__project-rag__search_project",
        "mcp__project-rag__open_source",
        "mcp__project-rag__get_ingestion_status",
    ]
