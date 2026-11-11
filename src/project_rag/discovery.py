"""Recursive discovery, hashing, and supported-format classification (v1 §4.1)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from pydantic import BaseModel

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md"})

# Temp/lock/derived/index artifacts to ignore (v1 §4.1).
IGNORE_SUFFIXES: tuple[str, ...] = (".tmp", ".lock", "~")
IGNORE_DIR_NAMES: frozenset[str] = frozenset(
    {".git", "local_data", "__pycache__", ".venv", "node_modules", "chroma"}
)
IGNORE_FILE_PREFIXES: tuple[str, ...] = ("~$", ".~")


class DiscoveredFile(BaseModel):
    source_path: str  # relative to root, forward-slash normalized
    abs_path: str
    ext: str
    size: int
    mtime: float  # diagnostic only (v1 §4.1)
    content_hash: str  # SHA-256
    supported: bool


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _long_path(path: str) -> str:
    """Prefix absolute Windows paths with \\\\?\\ so >260-char paths open."""
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path


def _is_ignored(path: Path) -> bool:
    name = path.name
    if any(name.startswith(p) for p in IGNORE_FILE_PREFIXES):
        return True
    return bool(any(name.endswith(s) for s in IGNORE_SUFFIXES))


def discover(root_path: str | Path) -> list[DiscoveredFile]:
    """Scan root recursively, hash contents, classify supported formats.

    Never discards unknown formats silently — they are returned with supported=False so
    the caller can record UNSUPPORTED (v1 §4.1).
    """
    root = Path(root_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"root_path is not a directory: {root}")

    # Long-path-safe walk: real corpora exceed Windows' 260-char MAX_PATH.
    long_root = _long_path(str(root))
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(long_root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIR_NAMES]
        found.extend(Path(dirpath, name) for name in filenames)

    out: list[DiscoveredFile] = []
    for p in sorted(found):
        if _is_ignored(p):
            continue
        ext = p.suffix.lower()
        rel = p.relative_to(long_root).as_posix()
        stat = p.stat()
        out.append(
            DiscoveredFile(
                source_path=rel,
                abs_path=str(p),
                ext=ext,
                size=stat.st_size,
                mtime=stat.st_mtime,
                content_hash=_sha256(p),
                supported=ext in SUPPORTED_EXTENSIONS,
            )
        )
    return out
