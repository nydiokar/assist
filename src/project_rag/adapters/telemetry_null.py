"""No-op telemetry (the prometheus adapter arrives with the service in S6)."""

from __future__ import annotations

from collections.abc import Mapping


class NullTelemetry:
    def observe(self, stage: str, seconds: float) -> None:
        return None

    def count(self, name: str, labels: Mapping[str, str] | None = None) -> None:
        return None

    def gauge(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        return None
