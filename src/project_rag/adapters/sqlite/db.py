"""The one SQLite store (spec A03 §15.2): connection factory, transactions, migrations.

- One connection per process, shared by the catalog, the FTS5 index and the sqlite-vec index, so a
  generation can be staged and published in ONE transaction.
- WAL journal (readers never block the writer), `foreign_keys=ON`, a busy timeout.
- Autocommit mode at the driver level; `transaction()` issues `BEGIN IMMEDIATE`/`COMMIT` itself and
  is re-entrant: an adapter write inside a service transaction joins it instead of committing.
- Migrations: numbered `migrations/NNN_name.sql` files applied in order, each in its own
  transaction that also sets `PRAGMA user_version = NNN`. Re-running is a no-op.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

from ...domain.errors import StoreError

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_MIGRATION_NAME = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")


def migration_files(directory: Path = MIGRATIONS_DIR) -> list[tuple[int, Path]]:
    """(version, path) for every migration file, sorted; versions must be 1..N without gaps."""
    found: list[tuple[int, Path]] = []
    for p in directory.iterdir():
        m = _MIGRATION_NAME.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort()
    expected = list(range(1, len(found) + 1))
    if [v for v, _ in found] != expected:
        raise StoreError(f"migration numbering must be 1..N without gaps: {[v for v, _ in found]}")
    return found


class SqliteDb:
    """Owns the connection. Not thread-safe by design: the service runs store access on one
    thread (spec §6.3)."""

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5000,
        migrations_dir: Path = MIGRATIONS_DIR,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._depth = 0
        self.vec_loaded = _try_load_sqlite_vec(self._conn)
        self.migrate(migrations_dir)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @property
    def path(self) -> Path:
        return self._path

    def user_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self, directory: Path = MIGRATIONS_DIR) -> list[int]:
        """Apply pending migrations; return the versions applied (empty when up to date)."""
        applied: list[int] = []
        current = self.user_version()
        for version, path in migration_files(directory):
            if version <= current:
                continue
            sql = path.read_text(encoding="utf-8")
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                for statement in _split_sql(sql):
                    self._conn.execute(statement)
                self._conn.execute(f"PRAGMA user_version = {version}")
                self._conn.execute("COMMIT")
            except sqlite3.Error as e:
                self._conn.execute("ROLLBACK")
                raise StoreError(f"migration {path.name} failed: {e}") from e
            applied.append(version)
        return applied

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Atomic unit of work; nested calls join the outermost transaction."""
        if self._depth > 0:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        self._conn.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield
        except BaseException:
            self._depth = 0
            self._conn.execute("ROLLBACK")
            raise
        self._depth = 0
        self._conn.execute("COMMIT")

    def close(self) -> None:
        self._conn.close()


def _split_sql(sql: str) -> list[str]:
    """Split a migration script on top-level semicolons (no triggers/procedures are used)."""
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def _try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Load the sqlite-vec extension when the package is installed (D-3 spike adapter)."""
    try:
        import sqlite_vec
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except (AttributeError, sqlite3.Error):
        return False
    finally:
        with suppress(AttributeError, sqlite3.Error):
            conn.enable_load_extension(False)
    return True
