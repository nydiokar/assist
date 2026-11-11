"""SQLite FTS5 BM25-like lexical channel (v1 §4.7, ADR-005).

Indexed from the same active chunks as the dense store. Project scope is a stored column
and filtering is enforced BEFORE returning results. Generation-scoped so stale chunks from a
superseded generation never surface. citation_text and provenance are carried alongside so
retrieval can build results without a second lookup.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ...domain.models import ChannelHit, Chunk, HitPayload, SourceLocation

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id UNINDEXED,
    project_id UNINDEXED,
    generation UNINDEXED,
    document_id UNINDEXED,
    source_path UNINDEXED,
    heading UNINDEXED,
    location_json UNINDEXED,
    citation_text UNINDEXED,
    doc_class UNINDEXED,
    body,
    tokenize = 'unicode61 remove_diacritics 0'
);
"""


def _escape_fts_query(query: str) -> str:
    # Wrap each term as a quoted phrase to neutralize FTS5 operators from user/LLM input.
    terms = [t for t in query.replace('"', " ").split() if t]
    if not terms:
        return '""'
    return " OR ".join(f'"{t}"' for t in terms)


class Fts5LexicalIndex:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_fts5()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _ensure_fts5(self) -> None:
        try:
            self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_probe USING fts5(x)")
            self._conn.execute("DROP TABLE IF EXISTS _fts_probe")
        except sqlite3.OperationalError as e:  # pragma: no cover
            raise RuntimeError("SQLite build lacks FTS5 support") from e

    def close(self) -> None:
        self._conn.close()

    def add_chunks(self, generation: str, chunks: list[Chunk]) -> None:
        self._conn.executemany(
            "INSERT INTO chunk_fts(chunk_id, project_id, generation, document_id, source_path, "
            "heading, location_json, citation_text, doc_class, body) VALUES(?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    c.chunk_id,
                    c.project_id,
                    generation,
                    c.document_id,
                    c.source_path,
                    c.heading or "",
                    c.location.model_dump_json(),
                    c.citation_text,
                    c.doc_class,
                    c.embed_text,
                )
                for c in chunks
            ],
        )
        self._conn.commit()

    def delete_generation(self, project_id: str, generation: str) -> None:
        self._conn.execute(
            "DELETE FROM chunk_fts WHERE project_id=? AND generation=?",
            (project_id, generation),
        )
        self._conn.commit()

    def search(
        self, project_id: str, generation: str, query: str, limit: int, doc_class: str | None = None
    ) -> list[ChannelHit]:
        """Return hits ordered by bm25 ascending (lower == better)."""
        fts_query = _escape_fts_query(query)
        rows = self._conn.execute(
            "SELECT chunk_id, document_id, source_path, heading, location_json, citation_text, "
            "bm25(chunk_fts) AS score FROM chunk_fts "
            "WHERE chunk_fts MATCH ? AND project_id=? AND generation=? "
            "AND (? IS NULL OR doc_class=?) "
            "ORDER BY score LIMIT ?",
            (fts_query, project_id, generation, doc_class, doc_class, limit),
        ).fetchall()
        return [
            ChannelHit(
                chunk_id=r["chunk_id"],
                score=float(r["score"]),
                payload=HitPayload(
                    document_id=r["document_id"],
                    source_path=r["source_path"],
                    heading=r["heading"],
                    location=SourceLocation.model_validate_json(r["location_json"]),
                    citation_text=r["citation_text"],
                ),
            )
            for r in rows
        ]
