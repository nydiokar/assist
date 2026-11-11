"""Writes each RunReport as `<dir>/<started_at>_<project>_<run_id>.json` (S4, spec §6.5)."""

from __future__ import annotations

import re
from pathlib import Path

from ..domain.models import RunReport


class JsonReportWriter:
    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)

    def write(self, report: RunReport) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        stamp = re.sub(r"[^0-9T]", "", report.started_at)[:15]
        name = f"{stamp}_{report.project_id}_{report.run_id}.json"
        (self._dir / name).write_text(report.model_dump_json(indent=1), encoding="utf-8")


class NullReportWriter:
    def write(self, report: RunReport) -> None:
        return None
