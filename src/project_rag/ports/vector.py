"""Vector index port (memory/chroma today; sqlite-vec candidate in S2, D-3)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from ..domain.models import ChannelHit, Chunk


class VectorIndex(Protocol):
    def add(
        self, generation: str, chunks: list[Chunk], embeddings: npt.NDArray[np.float32]
    ) -> None: ...

    def delete_generation(self, project_id: str, generation: str) -> None:
        """Remove exactly this generation's vectors. Chunk ids repeat across generations (an
        unchanged file keeps its ids), so there is deliberately no delete-by-chunk-id."""
        ...

    def query(
        self,
        project_id: str,
        generation: str,
        query_embedding: npt.NDArray[np.float32],
        limit: int,
        doc_class: str | None = None,
    ) -> list[ChannelHit]:
        """Nearest hits (cosine distance ascending), filtered to project + generation first."""
        ...

    def count(self, project_id: str, generation: str) -> int: ...
