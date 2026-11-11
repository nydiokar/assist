"""Telemetry port (spec §6.10). Implementations must never raise."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class Telemetry(Protocol):
    def observe(self, stage: str, seconds: float) -> None: ...

    def count(self, name: str, labels: Mapping[str, str] | None = None) -> None: ...

    def gauge(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None: ...
