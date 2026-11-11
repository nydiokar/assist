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

from .api import ProjectRAG
from .interfaces.mcp.tools import build_server

__all__ = ["build_server", "main"]


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
