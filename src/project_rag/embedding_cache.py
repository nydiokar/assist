"""Persistent embedding cache, stage [3] of A01 §C (ledger F13).

Key = (sha256(embed_text), embedding_profile.key()). A changed model/profile never hits old
vectors; an unchanged chunk is embedded once ever. Batches are committed as they finish, so an
interrupted multi-hour run resumes instead of restarting. Vectors are stored FP32.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .core.fingerprint import text_hash  # noqa: F401  (re-export, S1 shim)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS emb (
    profile TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (profile, text_hash)
) WITHOUT ROWID;
"""


class EmbeddingCache:
    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(p))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get_many(self, profile: str, hashes: list[str], dim: int) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(hashes), 500):
            part = hashes[i : i + 500]
            marks = ",".join("?" * len(part))
            rows = self._conn.execute(
                f"SELECT text_hash, vec FROM emb WHERE profile=? AND dim=? "
                f"AND text_hash IN ({marks})",
                (profile, dim, *part),
            ).fetchall()
            for h, blob in rows:
                found[h] = np.frombuffer(blob, dtype=np.float32).copy()
        return found

    def put_many(self, profile: str, items: dict[str, np.ndarray]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO emb(profile, text_hash, dim, vec) VALUES(?,?,?,?)",
            [
                (profile, h, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes())
                for h, v in items.items()
            ],
        )
        self._conn.commit()
