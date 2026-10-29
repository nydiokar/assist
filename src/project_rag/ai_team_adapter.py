"""AI-Team tool registration surface (v1 §3.3, ADR-001, v2 §D7; WP4 wiring 2026-10-29).

Exposes RAG's read-only tools to the AI-Team Claude-Agent-SDK session as an **in-process MCP
server** — the same mechanism AI-Team already runs for its `send_instruction` tool
(`src/control/agent_sender.py:build_claude_sender_server`). This routes through the Claude Code
SDK on the owner's existing plan — NO Anthropic API key, NO direct Messages-API egress. RAG is
bound as an MCP toolset, not a hosted-API client.

`build_rag_mcp_server(rag, project_id)` returns a `ClaudeAgentOptions.mcp_servers` value; AI-Team's
`claude_driver._sdk_options` merges it under `RAG_SERVER_NAME` and appends `RAG_TOOL_FQNS` to
`allowed_tools` — exactly as it does for the sender server. `claude_agent_sdk` is imported lazily so
this module (and the rest of project_rag) loads without that dependency.

project_id is resolved from verified application/session context BEFORE these are built — never
taken from the LLM (v1 §4.6/§5.1 step 1). Tools are read-only; no filesystem fallback is exposed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from .api import ProjectRAG

# MCP identity, mirroring agent_sender's naming (FQN = mcp__<server>__<tool>).
RAG_SERVER_NAME = "project_rag"
RAG_TOOL_NAMES = ("search_project", "open_source", "get_ingestion_status")
RAG_TOOL_FQNS = tuple(f"mcp__{RAG_SERVER_NAME}__{name}" for name in RAG_TOOL_NAMES)

# Grounding guidance the agent should follow when using these tools (v1 §2.2/§5.2). Surfaced in the
# search_project tool description so the behavior travels with the tool, whatever harness binds it.
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


def _as_tool_result(payload: object) -> dict:
    """Wrap a handler's dict result in the SDK MCP tool-result shape (content blocks + is_error),
    matching agent_sender's `render_outcome` contract."""
    text = json.dumps(payload, ensure_ascii=False, default=str)
    warnings = payload.get("warnings") if isinstance(payload, dict) else None
    is_error = bool(warnings) and any("not registered" in str(w) or "no active generation" in str(w)
                                      for w in warnings)
    return {"content": [{"type": "text", "text": text}], "is_error": is_error}


def build_rag_mcp_server(rag: ProjectRAG, project_id: str) -> dict:
    """In-process SDK MCP server exposing RAG's read-only tools (ClaudeAgentOptions.mcp_servers
    value). Mirrors agent_sender.build_claude_sender_server. `claude_agent_sdk` imported lazily.

    The server is bound to the server-resolved `project_id` (not the LLM's). Blocking RAG calls run
    off the SDK event loop via asyncio.to_thread, as the sender does for its HTTP call.
    """
    import asyncio  # noqa: PLC0415

    from claude_agent_sdk import create_sdk_mcp_server, tool  # noqa: PLC0415

    descriptors = {d.name: d for d in build_tool_descriptors(rag, project_id)}

    @tool("search_project",
          "Retrieve source-backed passages for the active project. " + GROUNDING_GUIDANCE,
          {"type": "object",
           "properties": {"query": {"type": "string"},
                          "top_k": {"type": "integer", "default": 5}},
           "required": ["query"]})
    async def _search(args: dict) -> dict:
        payload = await asyncio.to_thread(
            descriptors["search_project"].handler,
            args["query"], args.get("top_k", 5))
        return _as_tool_result(payload)

    @tool("open_source",
          "Return the exact original excerpt for a retrieved document/location to verify a span "
          "before citing a numeric or named fact.",
          {"type": "object",
           "properties": {"document_id": {"type": "string"},
                          "source_path": {"type": "string"},
                          "max_chars": {"type": "integer", "default": 6000}},
           "required": ["document_id", "source_path"]})
    async def _open(args: dict) -> dict:
        payload = await asyncio.to_thread(
            descriptors["open_source"].handler,
            args["document_id"], args["source_path"], args.get("max_chars", 6000))
        return _as_tool_result(payload)

    @tool("get_ingestion_status",
          "Report corpus ingestion health/coverage for the active project.",
          {"type": "object", "properties": {}})
    async def _status(_args: dict) -> dict:
        payload = await asyncio.to_thread(descriptors["get_ingestion_status"].handler)
        return _as_tool_result(payload)

    return dict(create_sdk_mcp_server(
        name=RAG_SERVER_NAME, version="1.0.0", tools=[_search, _open, _status]))


def register_with_ai_team(rag: ProjectRAG, project_id: str) -> list[ToolDescriptor]:
    """Return the raw descriptors. The live WP4 binding is `build_rag_mcp_server` above, merged by
    AI-Team's `claude_driver._sdk_options` into `mcp_servers={RAG_SERVER_NAME: ...}` with
    `RAG_TOOL_FQNS` appended to `allowed_tools` (mirrors the sender server). Kept for callers that
    want the descriptors without the SDK dependency."""
    return build_tool_descriptors(rag, project_id)
