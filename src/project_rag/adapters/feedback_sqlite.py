"""Feedback sink on the shared SQLite store (table `feedback`, migrations 005/006; spec §6.9)."""

from __future__ import annotations

from datetime import datetime

from ..domain.models import FeedbackSignal
from .sqlite.db import SqliteDb


class SqliteFeedbackSink:
    def __init__(self, db: SqliteDb) -> None:
        self._db = db

    def record(self, signal: FeedbackSignal) -> None:
        with self._db.transaction():
            self._db.conn.execute(
                "INSERT INTO feedback(kind, project_id, at, query_hash, detail, query_text) "
                "VALUES(?,?,?,?,?,?)",
                (
                    signal.kind,
                    signal.project_id,
                    signal.at.isoformat(),
                    signal.query_hash,
                    signal.detail,
                    signal.query,
                ),
            )

    def query(self, since: datetime) -> list[FeedbackSignal]:
        rows = self._db.conn.execute(
            "SELECT kind, project_id, at, query_hash, detail, query_text FROM feedback "
            "WHERE at >= ? ORDER BY at, id",
            (since.isoformat(),),
        ).fetchall()
        return [
            FeedbackSignal(
                kind=r["kind"],
                project_id=r["project_id"],
                at=datetime.fromisoformat(r["at"]),
                query_hash=r["query_hash"],
                detail=r["detail"],
                query=r["query_text"],
            )
            for r in rows
        ]
