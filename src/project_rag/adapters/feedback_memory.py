"""In-memory feedback sink (tests and dev). Signals are kept in arrival order."""

from __future__ import annotations

from datetime import datetime

from ..domain.models import FeedbackSignal


class MemoryFeedbackSink:
    def __init__(self) -> None:
        self._signals: list[FeedbackSignal] = []

    def record(self, signal: FeedbackSignal) -> None:
        self._signals.append(signal)

    def query(self, since: datetime) -> list[FeedbackSignal]:
        return sorted((s for s in self._signals if s.at >= since), key=lambda s: s.at)
