"""Migration runner (S2): empty DB -> latest, re-run is a no-op, legacy DBs open."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from project_rag.adapters.sqlite.db import MIGRATIONS_DIR, SqliteDb, migration_files
from project_rag.domain.errors import StoreError


def _tables(db: SqliteDb) -> set[str]:
    rows = db.conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table')").fetchall()
    return {r[0] for r in rows}


def test_empty_db_migrates_to_latest_and_rerun_is_noop(tmp_path: Path) -> None:
    n_files = len(list(MIGRATIONS_DIR.glob("*.sql")))
    assert n_files == len(migration_files())
    db = SqliteDb(tmp_path / "s.sqlite3")
    assert db.user_version() == n_files
    assert {"projects", "documents", "chunks", "chunk_fts", "staged_generations"} <= _tables(db)
    assert {"run_reports", "feedback"} <= _tables(db)
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(projects)").fetchall()}
    assert {"drive_id", "slug", "display_name"} <= cols
    assert db.migrate() == []  # no-op
    db.close()
    again = SqliteDb(tmp_path / "s.sqlite3")  # reopening applies nothing either
    assert again.user_version() == n_files
    again.close()


def test_pragmas(tmp_path: Path) -> None:
    db = SqliteDb(tmp_path / "s.sqlite3")
    assert db.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert db.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    db.close()


def test_prototype_database_opens_and_is_upgraded(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(path)  # the pre-S2 schema: no user_version, catalog tables exist
    conn.executescript(
        "CREATE TABLE projects (project_id TEXT PRIMARY KEY, active_generation TEXT, "
        "embedding_profile_key TEXT, collection TEXT);"
        "INSERT INTO projects VALUES ('ypq', 'g_old', 'k', 'c');"
    )
    conn.close()
    db = SqliteDb(path)
    assert db.user_version() == len(migration_files())
    row = db.conn.execute("SELECT active_generation, drive_id FROM projects").fetchone()
    assert tuple(row) == ("g_old", None)
    db.close()


def test_failed_migration_rolls_back_and_keeps_version(tmp_path: Path) -> None:
    mig = tmp_path / "migs"
    mig.mkdir()
    (mig / "001_ok.sql").write_text("CREATE TABLE a (x INTEGER);", encoding="utf-8")
    (mig / "002_bad.sql").write_text(
        "CREATE TABLE b (x INTEGER);\nTHIS IS NOT SQL;", encoding="utf-8"
    )
    with pytest.raises(StoreError, match="002_bad"):
        SqliteDb(tmp_path / "m.sqlite3", migrations_dir=mig)
    conn = sqlite3.connect(tmp_path / "m.sqlite3")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master").fetchall()}
    assert "a" in names and "b" not in names
    conn.close()


def test_migration_numbering_gap_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "001_a.sql").write_text("SELECT 1;", encoding="utf-8")
    (tmp_path / "003_c.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(StoreError, match="without gaps"):
        migration_files(tmp_path)


def test_nested_transaction_joins_outer_and_rolls_back_together(tmp_path: Path) -> None:
    db = SqliteDb(tmp_path / "t.sqlite3")
    db.conn.execute("CREATE TABLE t (x INTEGER)")
    with pytest.raises(RuntimeError), db.transaction():
        db.conn.execute("INSERT INTO t VALUES (1)")
        with db.transaction():
            db.conn.execute("INSERT INTO t VALUES (2)")
        raise RuntimeError("boom")
    assert db.conn.execute("SELECT count(*) FROM t").fetchone()[0] == 0
    with db.transaction():
        db.conn.execute("INSERT INTO t VALUES (3)")
    assert db.conn.execute("SELECT count(*) FROM t").fetchone()[0] == 1
    db.close()
