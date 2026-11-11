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
