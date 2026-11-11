"""Catalog port: projects, documents, generations, pinned embedding profile (v1 §4.3/§4.6)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..domain.models import DocumentRecord, EmbeddingProfile, ProjectIdentity, RunReport


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

    def active_generations(self) -> dict[str, str]:
        """{project_id: active generation} for every published project."""
        ...

    def publish_generation(self, project_id: str, generation: str) -> None:
        """Flip the active pointer (the publish commit point) and close the staging entry."""
        ...

    def prior_hashes(self, project_id: str, generation: str) -> dict[str, tuple[str, str]]:
        """{source_path: (content_hash, source_id)} for a generation."""
        ...

    def chunk_ids_for_generation(self, project_id: str, generation: str) -> list[str]: ...

    def document_states(self, project_id: str, generation: str) -> list[DocumentRecord]: ...

    def delete_generation(self, project_id: str, generation: str) -> None: ...

    def begin_staging(self, project_id: str, generation: str, at: datetime) -> None:
        """Journal a generation before any of its rows or vectors are written (two-phase)."""
        ...

    def stale_generations(self) -> list[tuple[str, str]]:
        """Non-active (project_id, generation) pairs: crashed stagings and unfinished deletes."""
        ...

    def project_by_drive_id(self, drive_id: str) -> ProjectIdentity | None: ...

    def list_projects(self) -> list[ProjectIdentity]:
        """Registered projects (rows with a drive_id), by project_id."""
        ...

    def upsert_project_identity(self, identity: ProjectIdentity) -> None: ...

    def all_project_ids(self) -> set[str]:
        """Every project_id in the catalog (registered or ingested directly)."""
        ...

    def record_run(self, report: RunReport) -> None:
        """Persist one ingest run report (table `run_reports`)."""
        ...

    def close(self) -> None: ...
