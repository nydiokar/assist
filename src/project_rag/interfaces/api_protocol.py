"""The surface the interfaces (CLI, MCP) drive. Implemented by the ProjectRAG facade over the
services; interfaces depend on this Protocol, never on wiring or adapters (R1)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from ..domain.models import (
    IngestReport,
    SearchResponse,
    SourceLocation,
    SourceResponse,
    StatusReport,
)


class AssistantAPI(Protocol):
    def register_project_root(self, project_id: str, root_path: str) -> None: ...

    def ingest_project(
        self, project_id: str, root_path: str, *, dry_run: bool = False
    ) -> IngestReport: ...

    def search_project(
        self,
        project_id: str,
        query: str,
        *,
        top_k: int | None = None,
        filters: Mapping[str, object] | None = None,
        as_of: str | None = None,
    ) -> SearchResponse: ...

    def open_source(
        self,
        project_id: str,
        document_id: str,
        *,
        source_path: str | None = None,
        location: SourceLocation | None = None,
        max_chars: int | None = None,
    ) -> SourceResponse: ...

    def get_ingestion_status(self, project_id: str) -> StatusReport: ...

    def close(self) -> None: ...
