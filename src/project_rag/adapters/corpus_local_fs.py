"""Local filesystem corpus reader (v1 §4.1): long-path-safe discovery, hashing, supported-format
classification, and path resolution confined to the project root (v1 §8)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ..domain.errors import PathOutsideRoot
from ..domain.models import DiscoveredFile

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md"})

# Temp/lock/derived/index artifacts to ignore (v1 §4.1).
IGNORE_SUFFIXES: tuple[str, ...] = (".tmp", ".lock", "~")
IGNORE_DIR_NAMES: frozenset[str] = frozenset(
    {".git", "local_data", "__pycache__", ".venv", "node_modules", "chroma"}
)
IGNORE_FILE_PREFIXES: tuple[str, ...] = ("~$", ".~")


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


def resolve_within_root(root: str | Path, source_path: str) -> Path:
    """Canonicalise `source_path` under `root`; raise PathOutsideRoot on traversal/symlink escape."""
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / source_path).resolve()
    if root_resolved != candidate and root_resolved not in candidate.parents:
        raise PathOutsideRoot(f"path escapes project root: {source_path}")
    return candidate


class LocalFsCorpus:
    """CorpusReader over the local filesystem."""

    def list_files(self, root: str) -> list[DiscoveredFile]:
        return discover(root)

    def resolve(self, root: str, source_path: str) -> str:
        return str(resolve_within_root(root, source_path))

    def is_file(self, path: str) -> bool:
        return Path(path).is_file()
