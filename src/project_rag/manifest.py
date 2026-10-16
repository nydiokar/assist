"""SQLite document state + build versions + collection compatibility (v1 §4.3/§4.6).

Stores per-source document state, the active index_generation per project, the pinned
embedding_profile and collection schema. Refuses writes on profile/schema mismatch (v2 §D5).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .schema import EmbeddingProfile

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    active_generation TEXT,
    embedding_profile_key TEXT,
    collection TEXT
);
CREATE TABLE IF NOT EXISTS documents (
    project_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    version_id TEXT NOT NULL,
    state TEXT NOT NULL,
    generation TEXT NOT NULL,
    n_chunks INTEGER DEFAULT 0,
    diagnostics TEXT DEFAULT '',
    PRIMARY KEY (project_id, source_id, generation)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    generation TEXT NOT NULL,
    PRIMARY KEY (chunk_id, generation)
);
CREATE INDEX IF NOT EXISTS idx_chunks_gen ON chunks(project_id, generation);
CREATE INDEX IF NOT EXISTS idx_docs_gen ON documents(project_id, generation);
"""


class Manifest:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---- compatibility guard (v2 §D5 / v1 §4.6) ----
    def check_compatibility(self, project_id: str, profile: EmbeddingProfile,
                            collection: str) -> list[str]:
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

    def set_project_profile(self, project_id: str, profile: EmbeddingProfile,
                            collection: str) -> None:
        self._conn.execute(
            "INSERT INTO projects(project_id, embedding_profile_key, collection) VALUES(?,?,?) "
            "ON CONFLICT(project_id) DO UPDATE SET embedding_profile_key=excluded.embedding_profile_key, "
            "collection=excluded.collection",
            (project_id, profile.key(), collection),
        )
        self._conn.commit()

    # ---- generation + document records ----
    def upsert_document(self, *, project_id: str, source_id: str, source_path: str,
                        content_hash: str, version_id: str, state: str, generation: str,
                        n_chunks: int, diagnostics: str = "") -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO documents(project_id, source_id, source_path, content_hash, "
            "version_id, state, generation, n_chunks, diagnostics) VALUES(?,?,?,?,?,?,?,?,?)",
            (project_id, source_id, source_path, content_hash, version_id, state, generation,
             n_chunks, diagnostics),
        )
        self._conn.commit()

    def record_chunks(self, project_id: str, generation: str,
                      chunk_refs: list[tuple[str, str, str]]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks(chunk_id, project_id, source_id, version_id, generation) "
            "VALUES(?,?,?,?,?)",
            [(cid, project_id, sid, vid, generation) for cid, sid, vid in chunk_refs],
        )
        self._conn.commit()

    def active_generation(self, project_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT active_generation FROM projects WHERE project_id=?", (project_id,)
        ).fetchone()
        return row["active_generation"] if row else None

    def publish_generation(self, project_id: str, generation: str) -> None:
        self._conn.execute(
            "INSERT INTO projects(project_id, active_generation) VALUES(?,?) "
            "ON CONFLICT(project_id) DO UPDATE SET active_generation=excluded.active_generation",
            (project_id, generation),
        )
        self._conn.commit()

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

    def document_states(self, project_id: str, generation: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM documents WHERE project_id=? AND generation=?",
            (project_id, generation),
        ).fetchall()

    def delete_generation(self, project_id: str, generation: str) -> None:
        self._conn.execute("DELETE FROM documents WHERE project_id=? AND generation=?",
                           (project_id, generation))
        self._conn.execute("DELETE FROM chunks WHERE project_id=? AND generation=?",
                           (project_id, generation))
        self._conn.commit()
