"""RAG SDK-MCP server (WP4 wiring). Pure/offline — stubs claude_agent_sdk so no dependency and no
paid call. Verifies the tools wrap the adapter handlers, carry grounding guidance, use the FQN
naming AI-Team merges, and are bound to the server-resolved project_id (not the LLM's).
"""

from __future__ import annotations

import asyncio
import sys
import types

from project_rag import ai_team_adapter as ada


class _FakeRAG:
    """Minimal ProjectRAG stand-in: records the project_id each tool is called with."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def search_project(self, project_id, query, *, top_k=8):
        self.calls.append(("search", project_id, query, top_k))
        return types.SimpleNamespace(model_dump=lambda: {"results": [], "query": query,
                                                         "project_id": project_id})

    def open_source(self, project_id, document_id, *, source_path=None, max_chars=6000):
        self.calls.append(("open", project_id, document_id, source_path))
        return types.SimpleNamespace(model_dump=lambda: {"document_id": document_id,
                                                         "warnings": []})

    def get_ingestion_status(self, project_id):
        self.calls.append(("status", project_id))
        return types.SimpleNamespace(model_dump=lambda: {"project_id": project_id})


def _install_fake_sdk(monkeypatch_tools: list):
    """Install a fake claude_agent_sdk whose @tool records (name, desc, schema, fn) and whose
    create_sdk_mcp_server returns the collected tool fns for inspection."""
    mod = types.ModuleType("claude_agent_sdk")

    def tool(name, description, schema):
        def deco(fn):
            monkeypatch_tools.append((name, description, schema, fn))
            return fn
        return deco

    def create_sdk_mcp_server(*, name, version, tools):
        return {"name": name, "version": version, "tools": tools}

    mod.tool = tool
    mod.create_sdk_mcp_server = create_sdk_mcp_server
    sys.modules["claude_agent_sdk"] = mod


def test_fqn_naming_matches_ai_team_merge_contract():
    assert ada.RAG_SERVER_NAME == "project_rag"
    assert ada.RAG_TOOL_FQNS == (
        "mcp__project_rag__search_project",
        "mcp__project_rag__open_source",
        "mcp__project_rag__get_ingestion_status",
    )


def test_build_server_wraps_handlers_and_binds_project(monkeypatch):
    collected: list = []
    _install_fake_sdk(collected)
    try:
        rag = _FakeRAG()
        server = ada.build_rag_mcp_server(rag, "ypq")
        assert server["name"] == "project_rag"
        names = [c[0] for c in collected]
        assert names == ["search_project", "open_source", "get_ingestion_status"]
        # grounding guidance travels with the search tool
        search_desc = collected[0][1]
        assert "Retrieval-first" in search_desc
        assert "Not established from accessible project documents" in search_desc

        # invoke the search tool fn — it must pass the SERVER-resolved project id, not any arg
        search_fn = collected[0][3]
        result = asyncio.run(search_fn({"query": "бюджет", "top_k": 3}))
        assert result["is_error"] is False
        assert "бюджет" in result["content"][0]["text"]
        assert rag.calls[0] == ("search", "ypq", "бюджет", 3)
    finally:
        sys.modules.pop("claude_agent_sdk", None)


def test_tool_result_flags_error_on_unregistered_project():
    payload = {"warnings": ["project root not registered server-side"]}
    assert ada._as_tool_result(payload)["is_error"] is True
    assert ada._as_tool_result({"results": [1], "warnings": []})["is_error"] is False
