"""SQLite catalog: document state, generations, pinned embedding profile, staging journal
(v1 §4.3/§4.6; v3 S2). Shares the SqliteDb connection with the FTS5 (and sqlite-vec) indexes so
staging + the active-pointer flip are one transaction. Every write joins an outer transaction
when one is open, else commits on its own.
"""

from __future__ import annotations

from datetime import datetime

from ...domain.models import DocumentRecord, EmbeddingProfile
from .db import SqliteDb


class SqliteCatalog:
    def __init__(self, db: SqliteDb) -> None:
        self._db = db
        self._conn = db.conn

    def close(self) -> None:
        """The SqliteDb owns the connection; closing is the composition root's job."""
        return None

    # ---- compatibility guard (v2 §D5 / v1 §4.6) ----
    def check_compatibility(
        self, project_id: str, profile: EmbeddingProfile, collection: str
    ) -> list[str]:
        row = self._conn.execute(
            "SELECT embedding_profile_key, collection FROM projects WHERE project_id=?",
            (project_id,),
        ).fetchone()
        if row is None:
            return []
        problems: list[str] = []
        if row["embedding_profile_key"] and row["embedding_profile_key"] != profile.key():
            problems.append(
                f"embedding profile mismatch: stored={row['embedding_profile_key']} "
                f"live={profile.key()} (re-index as new generation per ADR-009)"
            )
        if row["collection"] and row["collection"] != collection:
            problems.append(f"collection mismatch: stored={row['collection']} live={collection}")
        return problems

    def set_project_profile(
        self, project_id: str, profile: EmbeddingProfile, collection: str
    ) -> None:
        with self._db.transaction():
            self._conn.execute(
                "INSERT INTO projects(project_id, embedding_profile_key, collection) "
                "VALUES(?,?,?) ON CONFLICT(project_id) DO UPDATE SET "
                "embedding_profile_key=excluded.embedding_profile_key, "
                "collection=excluded.collection",
                (project_id, profile.key(), collection),
            )

    # ---- generation + document records ----
    def upsert_document(self, record: DocumentRecord) -> None:
        with self._db.transaction():
            self._conn.execute(
                "INSERT OR REPLACE INTO documents(project_id, source_id, source_path, "
                "content_hash, version_id, state, generation, n_chunks, diagnostics) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    record.project_id,
                    record.source_id,
                    record.source_path,
                    record.content_hash,
                    record.version_id,
                    record.state,
                    record.generation,
                    record.n_chunks,
                    record.diagnostics,
                ),
            )

    def record_chunks(
        self, project_id: str, generation: str, chunk_refs: list[tuple[str, str, str]]
    ) -> None:
        with self._db.transaction():
            self._conn.executemany(
                "INSERT OR REPLACE INTO chunks(chunk_id, project_id, source_id, version_id, "
                "generation) VALUES(?,?,?,?,?)",
                [(cid, project_id, sid, vid, generation) for cid, sid, vid in chunk_refs],
            )

    def active_generation(self, project_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT active_generation FROM projects WHERE project_id=?", (project_id,)
        ).fetchone()
        return row["active_generation"] if row else None

    def active_generations(self) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT project_id, active_generation FROM projects "
            "WHERE active_generation IS NOT NULL"
        ).fetchall()
        return {r["project_id"]: r["active_generation"] for r in rows}

    def publish_generation(self, project_id: str, generation: str) -> None:
        """Flip the active pointer and close the generation's staging journal entry."""
        with self._db.transaction():
            self._conn.execute(
                "INSERT INTO projects(project_id, active_generation) VALUES(?,?) "
                "ON CONFLICT(project_id) DO UPDATE SET "
                "active_generation=excluded.active_generation",
                (project_id, generation),
            )
            self._conn.execute(
                "DELETE FROM staged_generations WHERE project_id=? AND generation=?",
                (project_id, generation),
            )

    def prior_hashes(self, project_id: str, generation: str) -> dict[str, tuple[str, str]]:
        """Return {source_path: (content_hash, source_id)} for a generation."""
        rows = self._conn.execute(
            "SELECT source_path, content_hash, source_id FROM documents "
            "WHERE project_id=? AND generation=?",
            (project_id, generation),
        ).fetchall()
        return {r["source_path"]: (r["content_hash"], r["source_id"]) for r in rows}

    def chunk_ids_for_generation(self, project_id: str, generation: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE project_id=? AND generation=?",
            (project_id, generation),
        ).fetchall()
        return [r["chunk_id"] for r in rows]

    def document_states(self, project_id: str, generation: str) -> list[DocumentRecord]:
        rows = self._conn.execute(
            "SELECT * FROM documents WHERE project_id=? AND generation=?",
            (project_id, generation),
        ).fetchall()
        return [
            DocumentRecord(
                project_id=r["project_id"],
                source_id=r["source_id"],
                source_path=r["source_path"],
                content_hash=r["content_hash"],
                version_id=r["version_id"],
                state=r["state"],
                generation=r["generation"],
                n_chunks=r["n_chunks"] or 0,
                diagnostics=r["diagnostics"] or "",
            )
            for r in rows
        ]

    def delete_generation(self, project_id: str, generation: str) -> None:
        with self._db.transaction():
            for table in ("documents", "chunks", "staged_generations"):
                self._conn.execute(
                    f"DELETE FROM {table} WHERE project_id=? AND generation=?",
                    (project_id, generation),
                )

    # ---- two-phase publish journal + GC (S2) ----
    def begin_staging(self, project_id: str, generation: str, at: datetime) -> None:
        with self._db.transaction():
            self._conn.execute(
                "INSERT OR REPLACE INTO staged_generations(project_id, generation, started_at) "
                "VALUES(?,?,?)",
                (project_id, generation, at.isoformat()),
            )

    def stale_generations(self) -> list[tuple[str, str]]:
        """(project_id, generation) pairs that are not active: crashed staging runs (journal)
        and superseded generations whose deletion did not finish."""
        rows = self._conn.execute(
            "SELECT project_id, generation FROM staged_generations "
            "UNION SELECT project_id, generation FROM documents "
            "UNION SELECT project_id, generation FROM chunks"
        ).fetchall()
        active = self.active_generations()
        return sorted(
            (r["project_id"], r["generation"])
            for r in rows
            if active.get(r["project_id"]) != r["generation"]
        )
