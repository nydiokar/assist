"""Feedback sink port (spec §6.9). The service records signals; the digest queries them."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..domain.models import FeedbackSignal


class FeedbackSink(Protocol):
    def record(self, signal: FeedbackSignal) -> None: ...

    def query(self, since: datetime) -> list[FeedbackSignal]:
        """Signals recorded at or after `since`, oldest first."""
        ...
