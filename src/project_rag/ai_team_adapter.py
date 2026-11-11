"""RAG tool contract (v1 §3.3, ADR-001, v2 §D7).

Defines the read-only tool descriptors (name, description, JSON schema, bound handler) that RAG
exposes to any agent. The live transport is a standalone **stdio MCP server**
(`project_rag.mcp_server`), registered in a project's `.mcp.json` and loaded by Claude Code /
AI-Team at session start — no harness/backend code is modified. This module is the thin,
transport-agnostic contract the MCP server (and any other adapter) consumes.

project_id is resolved from verified application/session context BEFORE descriptors are built —
never taken from the LLM (v1 §4.6/§5.1 step 1). Tools are read-only; no filesystem fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .api import ProjectRAG

# MCP identity. Tool names are what the agent calls; the server name is the `.mcp.json` key.
RAG_SERVER_NAME = "project-rag"
RAG_TOOL_NAMES = ("search_project", "open_source", "get_ingestion_status")

# Grounding guidance the agent should follow when using these tools (v1 §2.2/§5.2). Surfaced in the
# search_project tool description so the behaviour travels with the tool, whatever harness binds it.
GROUNDING_GUIDANCE = (
    "Retrieval-first: every corpus claim MUST be grounded in a passage returned by this tool, cited "
    "by source_path. If the evidence verdict is 'insufficient' or no returned passage actually "
    "states the asked fact, reply 'Not established from accessible project documents' (v1 §5.2) "
    "rather than guessing — a topically-plausible passage that lacks the fact is NOT sufficient. Do "
    "not count or total across passages as authoritative; retrieval returns evidence, not a complete "
    "set. Preserve Bulgarian (Cyrillic) text exactly."
)


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
        return rag.open_source(
            project_id, document_id, source_path=source_path, max_chars=max_chars
        ).model_dump()

    def status() -> dict:
        return rag.get_ingestion_status(project_id).model_dump()

    return [
        ToolDescriptor(
            name="search_project",
            read_only=True,
            handler=search,
            description="Retrieve source-backed passages for the active project. "
            + GROUNDING_GUIDANCE,
            json_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        ),
        ToolDescriptor(
            name="open_source",
            read_only=True,
            handler=open_src,
            description="Return the exact original excerpt for a retrieved document/location "
            "to verify extraction or expand context before numeric claims.",
            json_schema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "source_path": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 6000},
                },
                "required": ["document_id", "source_path"],
            },
        ),
        ToolDescriptor(
            name="get_ingestion_status",
            read_only=True,
            handler=status,
            description="Report corpus ingestion health/coverage for the active project.",
            json_schema={"type": "object", "properties": {}},
        ),
    ]


def tool_result_is_error(payload: object) -> bool:
    """A handler payload signals an error when it carries a fatal warning (project not registered,
    or no active index generation). The stdio MCP server uses this to set tool-result is_error."""
    warnings = payload.get("warnings") if isinstance(payload, dict) else None
    return bool(warnings) and any(
        "not registered" in str(w) or "no active generation" in str(w) for w in warnings
    )
