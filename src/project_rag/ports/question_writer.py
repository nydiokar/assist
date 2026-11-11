"""QuestionWriter port (S7, spec §6.8.2 tier S): writes known-item questions from chunks and
plausible-but-absent no-answer probes. Adapters: `fixture` (deterministic, tests/CI) and
`anthropic` (D-7 owner key, budget cap, cost logged)."""

from __future__ import annotations

from typing import Protocol


class QuestionWriter(Protocol):
    def question_for(self, chunk_text: str, source_path: str) -> str:
        """One natural question answerable from this chunk alone."""
        ...

    def no_answer_probes(self, project_summary: str, n: int) -> list[str]:
        """n questions about plausible topics the project does NOT cover."""
        ...
