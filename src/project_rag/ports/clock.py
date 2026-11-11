"""Clock and id generation, injected so the core and tests stay deterministic (R5)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """Timezone-aware current time (UTC)."""
        ...


class IdGen(Protocol):
    def new_id(self) -> str: ...
