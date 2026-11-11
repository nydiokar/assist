"""Anthropic QuestionWriter for tier-S smoke (D-7: the owner's maintenance API key; S7).

Budget-capped and cost-logged: every call's token usage is priced and summed, and the writer
refuses further calls once the cap is reached (`BudgetExceeded`). Only chunk text from the
indexed project is sent (that egress is exactly what D-7 decides). Server-side refusal
fallbacks are on; a refused or empty answer yields no question rather than an invented one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import anthropic

from ..domain.errors import BudgetExceeded

log = logging.getLogger(__name__)

# $ per 1M tokens (input, output), from the Claude API pricing table (cached 2026-11-03).
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}

QUESTION_SYSTEM = (
    "You write one evaluation question for a document-retrieval test. Read the passage and "
    "write ONE natural question a project staff member might ask that this passage alone "
    "answers. Use the passage's language (Bulgarian or English). Output only the question."
)
PROBE_SYSTEM = (
    "You write questions for a retrieval test that the document archive CANNOT answer. Given a "
    "list of the archive's folder names, write plausible questions about topics a project like "
    "this might have but this archive does not cover. One question per line, nothing else."
)


@dataclass
class CostLedger:
    cap_usd: float
    spent_usd: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    log_lines: list[str] = field(default_factory=list)


class AnthropicQuestionWriter:
    def __init__(
        self,
        *,
        model: str = "claude-opus-5-5",
        budget_usd: float = 2.0,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._client = client or anthropic.Anthropic()
        self._model = model
        self.ledger = CostLedger(cap_usd=budget_usd)

    def _ask(self, system: str, user: str, max_tokens: int) -> str:
        if self.ledger.spent_usd >= self.ledger.cap_usd:
            raise BudgetExceeded(
                f"tier-S budget ${self.ledger.cap_usd:.2f} reached (${self.ledger.spent_usd:.4f})"
            )
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": user}],
        )
        price_in, price_out = PRICES.get(self._model, PRICES["claude-opus-5-5"])
        usage = response.usage
        cost = (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1e6
        led = self.ledger
        led.calls += 1
        led.tokens_in += usage.input_tokens
        led.tokens_out += usage.output_tokens
        led.spent_usd += cost
        line = (
            f"tier-S call {led.calls}: model={response.model} in={usage.input_tokens} "
            f"out={usage.output_tokens} cost=${cost:.5f} total=${led.spent_usd:.4f}"
        )
        led.log_lines.append(line)
        log.info(line)
        if response.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def question_for(self, chunk_text: str, source_path: str) -> str:
        return self._ask(QUESTION_SYSTEM, f"Passage (from {source_path}):\n\n{chunk_text}", 512)

    def no_answer_probes(self, project_summary: str, n: int) -> list[str]:
        text = self._ask(PROBE_SYSTEM, f"Folders: {project_summary}\nWrite {n} questions.", 1024)
        return [ln.strip(" -0123456789.") for ln in text.splitlines() if ln.strip()][:n]
