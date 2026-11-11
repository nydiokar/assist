"""Stage caches (A01 §C): parse/chunk artifacts and the embedding cache (ledger F13)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from ..domain.models import ChunkArtifact, ParseArtifact


class ArtifactStore(Protocol):
    """Parse artifacts are shared across projects; chunk artifacts are per project."""

    def get_parse(self, key: str) -> ParseArtifact | None: ...

    def put_parse(self, artifact: ParseArtifact) -> None: ...

    def get_chunk(self, project_id: str, key: str) -> ChunkArtifact | None: ...

    def put_chunk(self, project_id: str, artifact: ChunkArtifact) -> None: ...


class EmbeddingCache(Protocol):
    """Vectors keyed by (profile key, text hash); batches commit as they finish (resumable)."""

    def get_many(
        self, profile: str, hashes: list[str], dim: int
    ) -> dict[str, npt.NDArray[np.float32]]: ...

    def put_many(self, profile: str, items: dict[str, npt.NDArray[np.float32]]) -> None: ...

    def close(self) -> None: ...
