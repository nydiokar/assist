"""Thin AI-Team tool registration surface (v1 §3.3, ADR-001, v2 §D7).

This module deliberately does NOT import or assume any AI-Team API. It exposes read-only tool
callables and a registration-descriptor list. The actual binding to AI-Team's tool runner is
performed at WP4 once the real tool contract and sandbox boundaries are inspected (v1 Open-
decision 3). The `# WP4 SEAM` marker is where that binding goes.

Only search_project / open_source are registered as read-only tools; get_ingestion_status is
optional diagnostics. No filesystem fallback is exposed here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .api import ProjectRAG


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    description: str
    read_only: bool
    handler: Callable[..., object]
    json_schema: dict


def build_tool_descriptors(rag: ProjectRAG, project_id: str) -> list[ToolDescriptor]:
    """Return descriptors bound to a server-resolved project scope.

    project_id is resolved from verified application/session context BEFORE this call — it is
    not taken from the LLM (v1 §4.6/§5.1 step 1).
    """

    def search(query: str, top_k: int = 8) -> dict:
        return rag.search_project(project_id, query, top_k=top_k).model_dump()

    def open_src(document_id: str, source_path: str, max_chars: int = 6000) -> dict:
        return rag.open_source(project_id, document_id, source_path=source_path,
                               max_chars=max_chars).model_dump()

    def status() -> dict:
        return rag.get_ingestion_status(project_id).model_dump()

    return [
        ToolDescriptor(
            name="search_project", read_only=True, handler=search,
            description="Retrieve source-backed passages for the active project. "
                        "Every corpus claim must be grounded in results from this tool.",
            json_schema={"type": "object", "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 8}}, "required": ["query"]},
        ),
        ToolDescriptor(
            name="open_source", read_only=True, handler=open_src,
            description="Return the exact original excerpt for a retrieved document/location "
                        "to verify extraction or expand context before numeric claims.",
            json_schema={"type": "object", "properties": {
                "document_id": {"type": "string"},
                "source_path": {"type": "string"},
                "max_chars": {"type": "integer", "default": 6000}},
                "required": ["document_id", "source_path"]},
        ),
        ToolDescriptor(
            name="get_ingestion_status", read_only=True, handler=status,
            description="Report corpus ingestion health/coverage for the active project.",
            json_schema={"type": "object", "properties": {}},
        ),
    ]


def register_with_ai_team(rag: ProjectRAG, project_id: str) -> list[ToolDescriptor]:
    descriptors = build_tool_descriptors(rag, project_id)
    # WP4 SEAM: bind `descriptors` to AI-Team's actual tool runner here once its registration
    # and per-session permission mechanism are inspected (WP0). Confirm direct filesystem tools
    # can be disabled per session before claiming grounded-answer provenance (v1 §5.2).
    return descriptors
