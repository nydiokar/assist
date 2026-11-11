"""Anthropic tier-S writer with a fake client (no network, no cost): request shape, cost ledger,
budget cap, refusal handling. The live tier-S run is owner-gated (D-7, X5)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from project_rag.adapters.writer_anthropic import AnthropicQuestionWriter
from project_rag.domain.errors import BudgetExceeded


class _FakeMessages:
    def __init__(self, replies: list[tuple[str, str]]) -> None:
        self.replies = replies
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        stop, text = self.replies.pop(0)
        return SimpleNamespace(
            model=kwargs["model"],
            stop_reason=stop,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
            content=[SimpleNamespace(type="text", text=text)] if text else [],
        )


def _writer(
    replies: list[tuple[str, str]], budget: float = 1.0
) -> tuple[AnthropicQuestionWriter, _FakeMessages]:
    fake = _FakeMessages(replies)
    client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    return AnthropicQuestionWriter(budget_usd=budget, client=client), fake  # type: ignore[arg-type]


def test_question_request_shape_and_cost_ledger() -> None:
    w, fake = _writer([("end_turn", "Какъв е бюджетът на проекта?")])
    assert w.question_for("Бюджет: 49 697 EUR", "proposal.docx") == "Какъв е бюджетът на проекта?"
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"] == {"effort": "low"}
    assert "proposal.docx" in str(call["messages"])
    # 1000 in x $4/M + 100 out x $20/M = $0.006
    assert abs(w.ledger.spent_usd - 0.006) < 1e-9 and w.ledger.calls == 1
    assert "cost=$0.00600" in w.ledger.log_lines[0]


def test_refusal_gives_no_question() -> None:
    w, _ = _writer([("refusal", "")])
    assert w.question_for("x", "a.md") == ""


def test_budget_cap_stops_calls() -> None:
    w, fake = _writer([("end_turn", "q1?"), ("end_turn", "q2?"), ("end_turn", "q3?")], budget=0.011)
    w.question_for("a", "a.md")
    w.question_for("b", "b.md")  # spent 0.012 >= cap
    with pytest.raises(BudgetExceeded):
        w.question_for("c", "c.md")
    assert len(fake.calls) == 2


def test_probes_are_split_per_line() -> None:
    w, _ = _writer([("end_turn", "1. Who catered?\n2. What was the hotel?\n3. Which bus company?")])
    assert w.no_answer_probes("Communications, Webinars", 3) == [
        "Who catered?",
        "What was the hotel?",
        "Which bus company?",
    ]
