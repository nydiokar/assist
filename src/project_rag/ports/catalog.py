"""Catalog port: projects, documents, generations, pinned embedding profile (v1 §4.3/§4.6)."""

from __future__ import annotations

from typing import Protocol

from ..domain.models import DocumentRecord, EmbeddingProfile


class Catalog(Protocol):
    def check_compatibility(
        self, project_id: str, profile: EmbeddingProfile, collection: str
    ) -> list[str]:
        """Problems that forbid writing with this profile/collection ([] = compatible)."""
        ...

    def set_project_profile(
        self, project_id: str, profile: EmbeddingProfile, collection: str
    ) -> None: ...

    def upsert_document(self, record: DocumentRecord) -> None: ...

    def record_chunks(
        self, project_id: str, generation: str, chunk_refs: list[tuple[str, str, str]]
    ) -> None:
        """Record (chunk_id, source_id, version_id) membership of a generation."""
        ...

    def active_generation(self, project_id: str) -> str | None: ...

    def publish_generation(self, project_id: str, generation: str) -> None: ...

    def prior_hashes(self, project_id: str, generation: str) -> dict[str, tuple[str, str]]:
        """{source_path: (content_hash, source_id)} for a generation."""
        ...

    def chunk_ids_for_generation(self, project_id: str, generation: str) -> list[str]: ...

    def document_states(self, project_id: str, generation: str) -> list[DocumentRecord]: ...

    def delete_generation(self, project_id: str, generation: str) -> None: ...

    def close(self) -> None: ...
