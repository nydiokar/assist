"""Run-report sink (S4): one JSON document per ingest run, kept next to the state."""

from __future__ import annotations

from typing import Protocol

from ..domain.models import RunReport


class ReportSink(Protocol):
    def write(self, report: RunReport) -> None: ...
