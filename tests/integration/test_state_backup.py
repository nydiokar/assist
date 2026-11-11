"""S8 exit check: backup -> restore round-trips the state DB (feedback, run reports, registry,
catalog) into a scratch directory with identical row counts and a clean integrity check."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from project_rag.adapters.feedback_sqlite import SqliteFeedbackSink
from project_rag.domain.config import Config, with_overrides
from project_rag.domain.models import FeedbackSignal
from project_rag.wiring import build
from tests.conftest import FIXTURE_CORPUS

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "state_backup.py"


def _run(*args: str) -> dict[str, object]:
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def test_backup_restore_round_trip(tmp_path: Path) -> None:
    state = tmp_path / "state"
    cfg = with_overrides(
        Config(),
        {"lexical_store.path": str(state / "store.sqlite3"), "vector_store.persist_path": ""},
    )
    s = build(cfg)
    s.ingest.ingest("demo", str(FIXTURE_CORPUS))
    SqliteFeedbackSink(s.db).record(
        FeedbackSignal(
            kind="rescue", project_id="demo", at=datetime.now(UTC), query="q", detail="x.md"
        )
    )
    s.close()

    backed = _run("backup", "--state", str(state), "--out", str(tmp_path / "bk"))
    counts = backed["counts"]
    assert isinstance(counts, dict) and counts["feedback"] == 1 and counts["run_reports"] == 1
    assert counts["chunks"] > 0 and backed["run_reports_json"] == 1
    restored = _run("restore", "--src", str(tmp_path / "bk"), "--into", str(tmp_path / "scratch"))
    assert restored["integrity"] == "ok" and restored["counts"] == counts
    assert _run("verify", "--a", str(state), "--b", str(tmp_path / "scratch"))["equal"] is True
