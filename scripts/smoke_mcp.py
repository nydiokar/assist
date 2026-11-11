"""End-to-end MCP smoke test over Streamable HTTP (S6): connect to a running assist service with
the bearer token, list the tools, list projects, run one retrieval, print the enveloped result.

  ASSIST_MCP_TOKEN=... uv run python scripts/smoke_mcp.py --url http://127.0.0.1:8000/mcp \
      [--project youth-participation] [--query "Какъв е общият бюджет на проекта?"]

`--fixture` starts an in-process service on the synthetic fixture corpus first (hash embedder,
no models), so the smoke runs anywhere.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

EXPECTED_TOOLS = [
    "get_ingestion_status",
    "list_projects",
    "open_source",
    "record_feedback",
    "search_all",
    "search_project",
]


async def run(url: str, token: str, project: str | None, query: str) -> int:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with (
        httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=120) as http,
        streamable_http_client(url, http_client=http) as streams,
        ClientSession(streams[0], streams[1]) as session,
    ):
        await session.initialize()
        names = sorted(t.name for t in (await session.list_tools()).tools)
        print("TOOLS:", names)
        if names != EXPECTED_TOOLS:
            print("SMOKE: FAIL (tool list)")
            return 1
        projects = json.loads((await session.call_tool("list_projects", {})).content[0].text)[
            "projects"
        ]
        print("PROJECTS:", [(p["project_id"], p["documents"]) for p in projects])
        pid = project or (projects[0]["project_id"] if projects else "")
        res = await session.call_tool(
            "search_project", {"project_id": pid, "query": query, "top_k": 5}
        )
        text = res.content[0].text
        if res.is_error:
            print("SMOKE: FAIL", text)
            return 1
        payload = json.loads(text)
        passages = payload["passages"]
        ev = payload["evidence"]
        print(f"QUERY: {query!r} project={pid} degraded={payload['degraded']}")
        print(f"  passages={len(passages)} sufficient={ev['sufficient']} top={ev['top_score']}")
        for p in passages[:3]:
            print(f"    [{p['rank']}] {p['trust']} {p['source_path']}")
        ok = bool(passages) and all(p["trust"] == "untrusted_data" for p in passages)
        print("SMOKE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1


def _fixture_service() -> tuple[str, str, object]:
    """Start the real service in-process on the fixture corpus; return (url, token, runtime)."""
    import shutil
    import socket
    import tempfile
    import threading
    import time

    import uvicorn

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from project_rag.domain.config import Config, with_overrides
    from project_rag.runtime import build_runtime

    tmp = Path(tempfile.mkdtemp(prefix="smoke_mcp_"))
    shutil.copytree(repo / "tests/fixtures/synthetic/ngo_project_demo", tmp / "corpus" / "Demo")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cfg = with_overrides(
        Config(),
        {
            "lexical_store.path": str(tmp / "state" / "store.sqlite3"),
            "vector_store.provider": "sqlite_vec",
            "vector_store.persist_path": "",
            "service.corpus_root": str(tmp / "corpus"),
            "service.mcp_port": port,
        },
    )
    os.environ["ASSIST_ALLOW_DEV_PROFILE"] = "1"
    rt = build_runtime(cfg, token="smoke")
    rt.worker.trigger()
    rt.worker.join(120)
    rt.executor.submit_sync(rt.api.sync_registry)
    server = uvicorn.Server(
        uvicorn.Config(rt.public_app, host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}/mcp", "smoke", rt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/mcp")
    ap.add_argument("--project", default=None)
    ap.add_argument("--query", default="Какъв е общият бюджет на проекта?")
    ap.add_argument("--fixture", action="store_true")
    args = ap.parse_args()
    url, token = args.url, os.environ.get("ASSIST_MCP_TOKEN", "")
    if args.fixture:
        url, token, _rt = _fixture_service()
        args.query = "What grant number is the Green Villages project under?"
    raise SystemExit(asyncio.run(run(url, token, args.project, args.query)))


if __name__ == "__main__":
    main()
