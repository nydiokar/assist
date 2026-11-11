"""Deterministic QuestionWriter for tests and CI (no LLM). A known-item question reuses the
chunk's own distinctive words; probes ask about topics that cannot be in an NGO archive."""

from __future__ import annotations

import re

_PROBES = [
    "What was the quarterly revenue of the satellite launch subsidiary?",
    "Which vaccine trial phase did the cardiology department complete?",
    "How many kilometres of railway track were electrified in 2019?",
    "What is the torque rating of the turbine gearbox?",
    "Who won the chess olympiad final mentioned in the minutes?",
]


class FixtureQuestionWriter:
    def question_for(self, chunk_text: str, source_path: str) -> str:
        words = [w for w in re.findall(r"\w+", chunk_text) if len(w) > 3][:8]
        return "What does the document say about " + " ".join(words) + "?"

    def no_answer_probes(self, project_summary: str, n: int) -> list[str]:
        return _PROBES[:n]
