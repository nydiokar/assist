"""Corpus reader port: lists a project tree and resolves paths safely inside its root."""

from __future__ import annotations

from typing import Protocol

from ..domain.models import DiscoveredFile


class CorpusReader(Protocol):
    def list_files(self, root: str) -> list[DiscoveredFile]:
        """Every file under `root` (ignored temp/lock files excluded), sorted, hashed."""
        ...

    def resolve(self, root: str, source_path: str) -> str:
        """Absolute path of `source_path` inside `root`. Raises PathOutsideRoot on escape."""
        ...

    def is_file(self, path: str) -> bool: ...

    def list_dirs(self, root: str) -> list[str]:
        """Top-level folder names under `root`, sorted; hidden (dot) folders excluded."""
        ...

    def read_text(self, root: str, source_path: str) -> str | None:
        """UTF-8 text of a file inside `root`, or None when it does not exist."""
        ...
