"""sqlite-vec VectorIndex (D-3 spike, S2). Lives in the shared SqliteDb so vectors are staged and
published in the same transaction as FTS5 and the catalog.

Layout: one vec0 table per dimension, `chunk_vec_d{dim}` (vec0 needs a fixed dimension), with
`project_id` as partition key, `generation`/`doc_class` as metadata columns (filtered inside the
KNN scan) and cosine distance. A companion table `chunk_vec_map` maps rowid <-> chunk_id and holds
the hit payload, so deletes by chunk id and payload reads need no vec0 auxiliary-column scan.

Not the chosen D-3 store (KNN p95 163 ms Linux / 318 ms Windows at 50k x 2560, over the 100 ms
bar; build plan §3). Kept as a contract-passing adapter so the decision stays reversible.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ...domain.errors import StoreError
from ...domain.models import ChannelHit, Chunk
from ..vector_payload import chunk_metadata, hit_payload
from .db import SqliteDb

_MAP_DDL = """
CREATE TABLE IF NOT EXISTS chunk_vec_map (
    rowid INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    generation TEXT NOT NULL,
    dim INTEGER NOT NULL,
    meta_json TEXT NOT NULL
)
"""
_MAP_IDX = "CREATE INDEX IF NOT EXISTS idx_vec_map_pg ON chunk_vec_map(project_id, generation)"
_MAP_CHUNK_IDX = "CREATE INDEX IF NOT EXISTS idx_vec_map_chunk ON chunk_vec_map(chunk_id)"


class SqliteVecIndex:
    def __init__(self, db: SqliteDb) -> None:
        if not db.vec_loaded:
            raise StoreError("sqlite-vec extension is not loadable in this Python build")
        self._db = db
        self._conn = db.conn
        with self._db.transaction():
            self._conn.execute(_MAP_DDL)
            self._conn.execute(_MAP_IDX)
            self._conn.execute(_MAP_CHUNK_IDX)

    def _table(self, dim: int) -> str:
        name = f"chunk_vec_d{int(dim)}"
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING vec0("
            f"project_id text partition key, embedding float[{int(dim)}] distance_metric=cosine, "
            f"generation text, doc_class text)"
        )
        return name

    def add(
        self, generation: str, chunks: list[Chunk], embeddings: npt.NDArray[np.float32]
    ) -> None:
        mat = np.asarray(embeddings, dtype=np.float32)
        if len(chunks) != mat.shape[0]:
            raise StoreError(f"{len(chunks)} chunks but {mat.shape[0]} vectors")
        if not chunks:
            return
        dim = int(mat.shape[1])
        try:
            with self._db.transaction():
                table = self._table(dim)
                for c, v in zip(chunks, mat, strict=True):
                    cur = self._conn.execute(
                        "INSERT INTO chunk_vec_map(chunk_id, project_id, generation, dim, "
                        "meta_json) VALUES(?,?,?,?,?)",
                        (
                            c.chunk_id,
                            c.project_id,
                            generation,
                            dim,
                            json.dumps(chunk_metadata(c, generation), ensure_ascii=False),
                        ),
                    )
                    self._conn.execute(
                        f"INSERT INTO {table}(rowid, project_id, embedding, generation, doc_class) "
                        "VALUES(?,?,?,?,?)",
                        (cur.lastrowid, c.project_id, v.tobytes(), generation, c.doc_class),
                    )
        except sqlite3.Error as e:
            raise StoreError(f"sqlite-vec add failed: {e}") from e

    def _delete_rows(self, rows: Sequence[sqlite3.Row]) -> None:
        for r in rows:
            self._conn.execute(
                f"DELETE FROM chunk_vec_d{int(r['dim'])} WHERE rowid=?", (r["rowid"],)
            )
            self._conn.execute("DELETE FROM chunk_vec_map WHERE rowid=?", (r["rowid"],))

    def delete_generation(self, project_id: str, generation: str) -> None:
        with self._db.transaction():
            rows = self._conn.execute(
                "SELECT rowid, dim FROM chunk_vec_map WHERE project_id=? AND generation=?",
                (project_id, generation),
            ).fetchall()
            self._delete_rows(rows)

    def query(
        self,
        project_id: str,
        generation: str,
        query_embedding: npt.NDArray[np.float32],
        limit: int,
        doc_class: str | None = None,
    ) -> list[ChannelHit]:
        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        table = f"chunk_vec_d{int(q.shape[0])}"
        exists = self._conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone()
        if exists is None:
            return []
        sql = (
            f"SELECT v.rowid AS rowid, v.distance AS distance, m.chunk_id AS chunk_id, "
            f"m.meta_json AS meta_json FROM {table} v JOIN chunk_vec_map m ON m.rowid = v.rowid "
            "WHERE v.embedding MATCH ? AND v.k = ? AND v.project_id = ? AND v.generation = ?"
        )
        params: list[object] = [q.tobytes(), int(limit), project_id, generation]
        if doc_class is not None:
            sql += " AND v.doc_class = ?"
            params.append(doc_class)
        sql += " ORDER BY v.distance"
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as e:
            raise StoreError(f"sqlite-vec query failed: {e}") from e
        return [
            ChannelHit(
                chunk_id=r["chunk_id"],
                score=float(r["distance"]),
                payload=hit_payload(json.loads(r["meta_json"])),
            )
            for r in rows
        ]

    def count(self, project_id: str, generation: str) -> int:
        row = self._conn.execute(
            "SELECT count(*) FROM chunk_vec_map WHERE project_id=? AND generation=?",
            (project_id, generation),
        ).fetchone()
        return int(row[0])
