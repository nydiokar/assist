"""Stdio MCP server wiring (WP4 transport). Offline — a fake ProjectRAG, no GPU, no paid call.

Verifies the server lists the three read-only tools with grounding guidance, routes a call to the
right handler with the SERVER-resolved project_id (never the caller's), and surfaces a fatal payload
as an error. Skips cleanly if `mcp` is absent (it lives in the WSL GPU venv, not the Windows .venv).
"""

from __future__ import annotations

import asyncio
import json
import types

import pytest

pytest.importorskip("mcp", reason="mcp SDK only in the WSL GPU venv")


class _FakeRAG:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def search_project(self, project_id, query, *, top_k=8):
        self.calls.append(("search", project_id, query, top_k))
        return types.SimpleNamespace(
            model_dump=lambda: {
                "results": [],
                "query": query,
                "project_id": project_id,
                "warnings": [],
            }
        )

    def open_source(self, project_id, document_id, *, source_path=None, max_chars=6000):
        self.calls.append(("open", project_id, document_id, source_path))
        return types.SimpleNamespace(
            model_dump=lambda: {"document_id": document_id, "warnings": []}
        )

    def get_ingestion_status(self, project_id):
        self.calls.append(("status", project_id))
        return types.SimpleNamespace(model_dump=lambda: {"project_id": project_id, "warnings": []})


def _build(rag):
    from project_rag.mcp_server import build_server

    return build_server(rag, "ypq")


def _text(call_result) -> str:
    """Extract the text payload from an MCPServer CallToolResult."""
    return call_result.content[0].text


def test_lists_three_readonly_tools_with_grounding():
    server = _build(_FakeRAG())
    tools = asyncio.run(server.list_tools())
    names = [t.name for t in tools]
    assert names == ["search_project", "open_source", "get_ingestion_status"]
    search = next(t for t in tools if t.name == "search_project")
    assert "Retrieval-first" in search.description
    assert "Not established from accessible project documents" in search.description


def test_call_routes_to_handler_with_server_project_id():
    rag = _FakeRAG()
    server = _build(rag)
    result = asyncio.run(server.call_tool("search_project", {"query": "бюджет", "top_k": 3}))
    payload = json.loads(_text(result))
    assert payload["query"] == "бюджет"
    assert payload["project_id"] == "ypq"  # server-resolved, not from args
    assert rag.calls[0] == ("search", "ypq", "бюджет", 3)


def test_fatal_payload_raises_so_agent_sees_error():
    class _Unreg(_FakeRAG):
        def search_project(self, project_id, query, *, top_k=8):
            return types.SimpleNamespace(
                model_dump=lambda: {"warnings": ["project root not registered server-side"]}
            )

    server = _build(_Unreg())
    # MCPServer raises a ToolError up to the protocol layer (reported to the agent as a tool error);
    # the deliberately-raised ToolError carries the "not registered" reason.
    with pytest.raises(Exception, match="not registered"):
        asyncio.run(server.call_tool("search_project", {"query": "x"}))
