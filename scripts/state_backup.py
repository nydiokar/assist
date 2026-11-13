"""Back up / restore the assist state that is not rebuildable (spec §8.6): the SQLite store
(feedback, run reports, registry, catalog) via the online backup API, plus the run-report JSON
files. The vector/FTS index inside the store is rebuildable but comes along for free; Chroma is
not backed up (rebuild = re-index, R6). Restic upload is done by `make backup` on the host.

  python scripts/state_backup.py backup  --state /state --out /backup/2026-11-11
  python scripts/state_backup.py restore --src /backup/2026-11-11 --into /tmp/restore-check
  python scripts/state_backup.py verify  --a /state --b /tmp/restore-check
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

STORE = "store.sqlite3"
TABLES = ("projects", "documents", "chunks", "run_reports", "feedback", "staged_generations")


def backup(state: Path, out: Path) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{state / STORE}?mode=ro", uri=True)
    dst = sqlite3.connect(out / STORE)
    with dst:
        src.backup(dst)  # consistent snapshot even while assist writes (WAL)
    src.close()
    dst.close()
    runs = state / "runs"
    n_runs = 0
    if runs.is_dir():
        shutil.copytree(runs, out / "runs", dirs_exist_ok=True)
        n_runs = len(list(runs.glob("*.json")))
    summary = {"store": STORE, "run_reports_json": n_runs, "counts": _counts(out / STORE)}
    (out / "BACKUP.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def restore(src: Path, into: Path) -> dict[str, object]:
    into.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / STORE, into / STORE)
    if (src / "runs").is_dir():
        shutil.copytree(src / "runs", into / "runs", dirs_exist_ok=True)
    conn = sqlite3.connect(into / STORE)
    ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    if ok != "ok":
        raise SystemExit(f"restored store failed integrity_check: {ok}")
    return {"integrity": ok, "user_version": version, "counts": _counts(into / STORE)}


def _counts(db: Path) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {
            t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES if t in names
        }
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backup")
    b.add_argument("--state", required=True)
    b.add_argument("--out", required=True)
    r = sub.add_parser("restore")
    r.add_argument("--src", required=True)
    r.add_argument("--into", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--a", required=True)
    v.add_argument("--b", required=True)
    args = ap.parse_args()
    if args.cmd == "backup":
        print(json.dumps(backup(Path(args.state), Path(args.out))))
    elif args.cmd == "restore":
        print(json.dumps(restore(Path(args.src), Path(args.into))))
    else:
        a, b2 = _counts(Path(args.a) / STORE), _counts(Path(args.b) / STORE)
        print(json.dumps({"a": a, "b": b2, "equal": a == b2}))
        if a != b2:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
