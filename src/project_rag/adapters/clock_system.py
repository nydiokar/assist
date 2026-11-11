"""Clock and id adapters: `system` for production, `fixed` for deterministic tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at


class UuidIdGen:
    def new_id(self) -> str:
        return uuid.uuid4().hex


class SequentialIdGen:
    def __init__(self, prefix: str = "id") -> None:
        self._prefix = prefix
        self._n = 0

    def new_id(self) -> str:
        self._n += 1
        return f"{self._prefix}{self._n:06d}"
