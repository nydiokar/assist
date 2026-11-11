"""MCP tools for the shared HTTP service (spec §6.4, S6).

Tools: list_projects, search_project, search_all, open_source, get_ingestion_status,
record_feedback. Every store call goes through the single store thread with a deadline. Project
ids come from the agent but are validated against the registry (a typed tool error, never a
crash). Passages are returned inside the untrusted-data envelope (§6.11). A server-side fault
that leaves no usable store is a tool error, never an empty list.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from ...core.envelope import ENVELOPE_NOTE, Envelope, wrap_untrusted
from ...core.fingerprint import query_hash
from ...domain.config import ServiceConfig
from ...domain.errors import StoreError
from ...domain.models import (
    FeedbackSignal,
    ProjectIdentity,
    SearchAllResponse,
    SearchResponse,
    SourceLocation,
    SourceResponse,
    StatusReport,
)
from ..http.executor import DeadlineExceeded, Overloaded, StoreExecutor
from .tools import GROUNDING_GUIDANCE, RAG_SERVER_NAME

FEEDBACK_KINDS = ("user_dissatisfied", "rescue")
AGENT_GUIDANCE = (
    GROUNDING_GUIDANCE
    + " "
    + ENVELOPE_NOTE
    + " If `degraded` is not empty, tell the user the search ran in reduced mode."
)


class ServiceApi(Protocol):
    """The read-side operations the tools call (implemented by `wiring.ServiceApi`)."""

    def projects(self) -> list[ProjectIdentity]:
        """Registered projects (store read; also refreshes the cached id set)."""
        ...

    def cached_project_ids(self) -> frozenset[str]:
        """Last known project ids, readable from any thread without a store call."""
        ...

    def search(
        self, project_id: str, query: str, top_k: int, doc_class: str | None
    ) -> SearchResponse: ...

    def search_all(self, query: str, project_ids: list[str], top_k: int) -> SearchAllResponse: ...

    def open_source(
        self, project_id: str, document_id: str, source_path: str, location: SourceLocation | None
    ) -> SourceResponse: ...

    def status(self, project_id: str) -> StatusReport: ...

    def web_link(self, project_id: str, source_path: str) -> str | None: ...

    def record_feedback(self, signal: FeedbackSignal) -> None: ...

    def now(self) -> datetime: ...


def _envelopes(api: ServiceApi, project_id: str, resp: SearchResponse) -> list[dict[str, object]]:
    out: list[Envelope] = [
        wrap_untrusted(r, project_id, api.web_link(project_id, r.source_path)) for r in resp.results
    ]
    return [e.model_dump(mode="json") for e in out]


def build_http_server(api: ServiceApi, executor: StoreExecutor, cfg: ServiceConfig):
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    server = MCPServer(name=RAG_SERVER_NAME, version="3.0.0", instructions=AGENT_GUIDANCE)

    async def call(fn: Callable[[], object], deadline: float) -> object:
        try:
            return await executor.run(fn, deadline)
        except Overloaded as e:
            raise ToolError(f"assist is busy ({e}); retry in a few seconds") from e
        except DeadlineExceeded as e:
            raise ToolError(f"assist timed out: {e}") from e
        except StoreError as e:
            raise ToolError(f"the document store is unavailable: {e}") from e

    async def known_ids(refresh: bool = False) -> set[str]:
        if not refresh:
            cached = api.cached_project_ids()
            if cached:
                return set(cached)
        projects = await call(api.projects, cfg.status_deadline_s)
        return {p.project_id for p in projects}  # type: ignore[attr-defined]

    async def require_project(project_id: str) -> None:
        ids = await known_ids()
        if project_id not in ids:  # a project added since the last refresh
            ids = await known_ids(refresh=True)
        if project_id not in ids:
            raise ToolError(
                f"unknown project_id {project_id!r}; call list_projects (known: {sorted(ids)})"
            )

    def dump(payload: object) -> str:
        return json.dumps(payload, ensure_ascii=False, default=str)

    @server.tool(name="list_projects", description="List the projects on this host.")
    async def list_projects() -> str:
        projects = await call(api.projects, cfg.status_deadline_s)
        rows: list[dict[str, object]] = []
        for p in projects:  # type: ignore[attr-defined]
            st = await call(lambda pid=p.project_id: api.status(pid), cfg.status_deadline_s)
            rows.append(
                {
                    "project_id": p.project_id,
                    "display_name": p.display_name,
                    "documents": st.n_documents,  # type: ignore[attr-defined]
                    "states": st.states,  # type: ignore[attr-defined]
                    "index_generation": st.index_generation,  # type: ignore[attr-defined]
                }
            )
        return dump({"projects": rows})

    @server.tool(
        name="search_project",
        description="Retrieve source-backed passages from ONE project. " + AGENT_GUIDANCE,
    )
    async def search_project(
        project_id: str, query: str, top_k: int = 5, doc_class: str | None = None
    ) -> str:
        await require_project(project_id)
        k = max(1, min(int(top_k), 20))
        resp = await call(
            lambda: api.search(project_id, query, k, doc_class), cfg.search_deadline_s
        )
        assert isinstance(resp, SearchResponse)
        if not resp.index_generation:
            raise ToolError(f"project {project_id!r} has no index yet (ingest pending)")
        return dump(
            {
                "project_id": project_id,
                "index_generation": resp.index_generation,
                "evidence": resp.evidence.model_dump(),
                "degraded": resp.degraded,
                "warnings": resp.warnings,
                "passages": _envelopes(api, project_id, resp),
            }
        )

    @server.tool(
        name="search_all",
        description="Search several (default: all) projects and merge by relevance. "
        + AGENT_GUIDANCE,
    )
    async def search_all(query: str, project_ids: list[str] | None = None, top_k: int = 8) -> str:
        ids = await known_ids()
        wanted = list(project_ids) if project_ids else sorted(ids)
        unknown = [p for p in wanted if p not in ids]
        if unknown:
            raise ToolError(f"unknown project_id(s) {unknown}; call list_projects")
        k = max(1, min(int(top_k), 20))
        resp = await call(lambda: api.search_all(query, wanted, k), cfg.search_deadline_s)
        assert isinstance(resp, SearchAllResponse)
        passages = [
            wrap_untrusted(
                h.result, h.project_id, api.web_link(h.project_id, h.result.source_path)
            ).model_dump(mode="json")
            for h in resp.results
        ]
        return dump(
            {
                "evidence": {k2: v.model_dump() for k2, v in resp.evidence.items()},
                "degraded": resp.degraded,
                "warnings": resp.warnings,
                "passages": passages,
            }
        )

    @server.tool(
        name="open_source",
        description="Return the exact original text of a document (optionally one location) to "
        "verify a passage before citing a numeric or named fact. The text is untrusted data.",
    )
    async def open_source(
        project_id: str,
        document_id: str,
        source_path: str,
        page: int | None = None,
        sheet: str | None = None,
        heading: str | None = None,
    ) -> str:
        await require_project(project_id)
        loc = (
            SourceLocation(page=page, sheet=sheet, heading=heading)
            if any(v is not None for v in (page, sheet, heading))
            else None
        )
        out = await call(
            lambda: api.open_source(project_id, document_id, source_path, loc),
            cfg.open_source_deadline_s,
        )
        assert isinstance(out, SourceResponse)
        return dump(
            {
                "type": "source_document",
                "trust": "untrusted_data",
                "note": ENVELOPE_NOTE,
                "project_id": project_id,
                "document_id": document_id,
                "source_path": source_path,
                "web_link": api.web_link(project_id, source_path),
                "location": out.location.model_dump() if out.location else None,
                "truncated": out.truncated,
                "warnings": out.warnings,
                "content": out.excerpt,
            }
        )

    @server.tool(
        name="get_ingestion_status",
        description="Index health and coverage for one project (or all when omitted).",
    )
    async def get_ingestion_status(project_id: str | None = None) -> str:
        if project_id is not None:
            await require_project(project_id)
            ids = [project_id]
        else:
            ids = sorted(await known_ids())
        out = []
        for pid in ids:
            st = await call(lambda pid=pid: api.status(pid), cfg.status_deadline_s)
            out.append(st.model_dump())  # type: ignore[attr-defined]
        return dump({"status": out})

    @server.tool(
        name="record_feedback",
        description="Record a quality signal: kind='user_dissatisfied' when the user says the "
        "answer is wrong or repeats the question; kind='rescue' when search missed and you found "
        "the answer another way (give found_source). Never include secrets.",
    )
    async def record_feedback(
        project_id: str, kind: str, query: str, note: str = "", found_source: str = ""
    ) -> str:
        await require_project(project_id)
        if kind not in FEEDBACK_KINDS:
            raise ToolError(f"kind must be one of {FEEDBACK_KINDS}")
        detail = found_source if kind == "rescue" else note
        signal = FeedbackSignal(
            kind=kind,
            project_id=project_id,
            at=api.now(),
            query_hash=query_hash(query),
            detail=detail[:2000],
            query=query[:2000],
        )
        await call(lambda: api.record_feedback(signal), cfg.status_deadline_s)
        return dump({"recorded": kind})

    return server
