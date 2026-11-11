"""In-process compatibility facade over the services (spec A03 §15.3 move rules).

`ProjectRAG` keeps the prototype's public surface (ingest_project, search_project, open_source,
get_ingestion_status, register_project_root, traces, close) for `scripts/` and the tests. It holds
no logic: everything is built by `wiring.build` and delegated to IngestService / SearchService.
It is removed once the scripts call the services directly.
"""

from __future__ import annotations

from collections.abc import Mapping

from .domain.config import Config
from .domain.models import (
    IngestReport,
    QueryTrace,
    SearchResponse,
    SourceLocation,
    SourceResponse,
    StatusReport,
)
from .wiring import Services, build, load_config


class ProjectRAG:
    def __init__(self, config: Config | None = None, config_path: str | None = None) -> None:
        self._services = build(config or load_config(config_path))

    @property
    def config(self) -> Config:
        return self._services.config

    @property
    def services(self) -> Services:
        return self._services

    def register_project_root(self, project_id: str, root_path: str) -> None:
        self._services.search.register_project_root(project_id, root_path)

    # ---- tool contracts ----

    def ingest_project(
        self, project_id: str, root_path: str, *, dry_run: bool = False
    ) -> IngestReport:
        self.register_project_root(project_id, root_path)
        return self._services.ingest.ingest(project_id, root_path, dry_run=dry_run)

    def search_project(
        self,
        project_id: str,
        query: str,
        *,
        top_k: int | None = None,
        filters: Mapping[str, object] | None = None,
        as_of: str | None = None,
    ) -> SearchResponse:
        return self._services.search.search_project(
            project_id, query, top_k=top_k, filters=filters, as_of=as_of
        )

    def open_source(
        self,
        project_id: str,
        document_id: str,
        *,
        source_path: str | None = None,
        location: SourceLocation | None = None,
        max_chars: int | None = None,
    ) -> SourceResponse:
        return self._services.search.open_source(
            project_id,
            document_id,
            source_path=source_path,
            location=location,
            max_chars=max_chars,
        )

    def get_ingestion_status(self, project_id: str) -> StatusReport:
        return self._services.ingest.status(project_id)

    # ---- diagnostics (grounding gate support, v1 §5.2) ----
    @property
    def traces(self) -> list[QueryTrace]:
        return self._services.search.traces

    def close(self) -> None:
        self._services.close()
