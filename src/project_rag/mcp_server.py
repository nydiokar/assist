"""Standalone stdio MCP server exposing project-RAG retrieval as agent tools.

This is the live WP4 transport. A thin adapter over the existing retrieval API (the RAG engine is
untouched and transport-agnostic, so Claude Code, Codex, or any MCP client reuses the same
retriever). Registered in a project's `.mcp.json` (`claude mcp add --scope project ...`) and loaded
at session start — NO harness/backend code modified, NO Anthropic API key, NO hosted-API egress.
The agent calls these tools and grounds on the returned passages.

Tools (read-only): search_project, open_source, get_ingestion_status. Project scope
(project_id + root) is bound here from launch args — never taken from the LLM (v1 §4.6).

Run (bound to one project; absolute paths — the agent's cwd is its own repo, not assist):
  python -m project_rag.mcp_server --config CONFIG.yaml --project PID --root /abs/path/to/corpus

The retriever is built once and kept warm for the server's lifetime (cold start ~30 s on GPU).
"""

from __future__ import annotations

import argparse
import json

from .ai_team_adapter import GROUNDING_GUIDANCE, RAG_SERVER_NAME, tool_result_is_error
from .api import ProjectRAG


def build_server(rag: ProjectRAG, project_id: str):
    """Construct the MCPServer bound to a warm ProjectRAG + server-resolved project_id.

    `project_id` is captured in the closures below — the LLM never supplies it (v1 §4.6). Calls run
    synchronously in the handler: a stdio MCP server serves ONE client serially, so there is no
    concurrency to gain by offloading, and ProjectRAG holds thread-affine SQLite connections
    (manifest + lexical store) that must be used on the thread that created them — so we must NOT
    hop threads (asyncio.to_thread would raise SQLite's cross-thread error). A fatal payload
    (project not registered, no active generation) is raised so the agent sees an MCP tool error,
    not a silent empty result.
    """
    from mcp.server import MCPServer

    try:  # a deliberately-raised ToolError keeps its text for the agent; a bare crash does not
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:  # pragma: no cover - fallback if the import path differs
        ToolError = RuntimeError  # type: ignore[assignment,misc]

    server = MCPServer(name=RAG_SERVER_NAME, version="1.0.0")

    def _dump(payload) -> str:
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if tool_result_is_error(payload):
            raise ToolError(text)
        return text

    @server.tool(
        name="search_project",
        description="Retrieve source-backed passages for the active project. " + GROUNDING_GUIDANCE,
    )
    async def search_project(query: str, top_k: int = 5) -> str:
        res = rag.search_project(project_id, query, top_k=top_k)
        return _dump(res.model_dump())

    @server.tool(
        name="open_source",
        description="Return the exact original excerpt for a retrieved document/location "
        "to verify a span before citing a numeric or named fact.",
    )
    async def open_source(document_id: str, source_path: str, max_chars: int = 6000) -> str:
        res = rag.open_source(project_id, document_id, source_path=source_path, max_chars=max_chars)
        return _dump(res.model_dump())

    @server.tool(
        name="get_ingestion_status",
        description="Report corpus ingestion health/coverage for the active project.",
    )
    async def get_ingestion_status() -> str:
        res = rag.get_ingestion_status(project_id)
        return _dump(res.model_dump())

    return server


def main() -> None:
    ap = argparse.ArgumentParser(description="project-RAG stdio MCP server")
    ap.add_argument("--config", required=True, help="retrieval config yaml (absolute path)")
    ap.add_argument("--project", required=True, help="server-resolved project_id")
    ap.add_argument("--root", required=True, help="project corpus root (absolute path)")
    args = ap.parse_args()

    rag = ProjectRAG(config_path=args.config)
    rag.register_project_root(args.project, args.root)
    server = build_server(rag, args.project)
    try:
        server.run(transport="stdio")
    finally:
        rag.close()


if __name__ == "__main__":
    main()
