"""End-to-end MCP smoke test: spawn the stdio server as a subprocess (exactly how Claude Code
launches it), connect an MCP client over stdio, list tools, and run a real retrieval against the
live index. Proves the wiring works before any agent is pointed at it.

Run in WSL (GPU index):
  PYTHONPATH=src python scripts/smoke_mcp.py --config configs/yp_qwen4b_gpu_report.yaml \
      --project ypq --root "$(pwd)/yspdb"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def run(config: str, project: str, root: str, query: str) -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "project_rag.mcp_server",
            "--config",
            config,
            "--project",
            project,
            "--root",
            root,
        ],
        env=None,
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        names = [t.name for t in tools.tools]
        print("TOOLS:", names)
        assert names == ["search_project", "open_source", "get_ingestion_status"], names

        res = await session.call_tool("search_project", {"query": query, "top_k": 5})
        # MCPServer may return the str via structuredContent and/or a content block; handle both
        text = None
        sc = getattr(res, "structuredContent", None)
        if isinstance(sc, dict):
            text = sc.get("result") if isinstance(sc.get("result"), str) else json.dumps(sc)
        if text is None and res.content:
            text = res.content[0].text
        payload = json.loads(text)
        n = len(payload.get("results", []))
        ev = payload.get("evidence", {})
        print(f"QUERY: {query!r}")
        print(
            f"  results={n}  evidence_sufficient={ev.get('sufficient')} "
            f"top_score={ev.get('top_score')}"
        )
        for r in payload.get("results", [])[:3]:
            print(f"    [{r['rank']}] {r['source_path']}")
            print(f"        {r['excerpt'][:140].strip()}")
        status = await session.call_tool("get_ingestion_status", {})
        st_text = (
            (status.structuredContent or {}).get("result")
            if getattr(status, "structuredContent", None)
            else status.content[0].text
        )
        st = json.loads(st_text)
        print(
            f"STATUS: n_documents={st.get('n_documents')} n_chunks={st.get('n_chunks')} "
            f"healthy={st.get('healthy')}"
        )
        ok = n > 0 and not res.is_error
        print("SMOKE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--project", default="ypq")
    ap.add_argument("--root", required=True)
    ap.add_argument("--query", default="Какъв е общият бюджет на проекта?")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args.config, args.project, args.root, args.query)))


if __name__ == "__main__":
    main()
