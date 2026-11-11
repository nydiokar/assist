"""Lexical index port (FTS5 today). Generation- and project-scoped."""

from __future__ import annotations

from typing import Protocol

from ..domain.models import ChannelHit, Chunk, ChunkView


class LexicalIndex(Protocol):
    def add_chunks(self, generation: str, chunks: list[Chunk]) -> None: ...

    def delete_generation(self, project_id: str, generation: str) -> None: ...

    def search(
        self,
        project_id: str,
        generation: str,
        query: str,
        limit: int,
        doc_class: str | None = None,
    ) -> list[ChannelHit]:
        """Hits best-first, filtered to project + generation (+ doc_class) before ranking."""
        ...

    def count(self, project_id: str, generation: str) -> int: ...

    def chunk_views(self, project_id: str, generation: str) -> list[ChunkView]:
        """Every chunk of a generation (id, source path, citation text), ordered by id."""
        ...

    def close(self) -> None: ...
