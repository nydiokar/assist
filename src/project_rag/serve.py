"""Run the assist service: public listener (/mcp, /healthz, /metrics) and admin listener
(/admin/*) in one process (spec §6.3, S6).

  ASSIST_MCP_TOKEN=... python -m project_rag.serve --config /config/deployment.yaml
"""

from __future__ import annotations

import argparse
import asyncio

import uvicorn

from .runtime import build_runtime
from .wiring import load_config


async def _serve(public_app: object, admin_app: object, cfg_service: object) -> None:
    svc = cfg_service
    public = uvicorn.Server(
        uvicorn.Config(public_app, host=svc.mcp_host, port=svc.mcp_port, log_level="info")  # type: ignore[attr-defined]
    )
    admin = uvicorn.Server(
        uvicorn.Config(admin_app, host=svc.admin_host, port=svc.admin_port, log_level="info")  # type: ignore[attr-defined]
    )
    await asyncio.gather(public.serve(), admin.serve())


def main() -> None:
    ap = argparse.ArgumentParser(description="assist service")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    runtime = build_runtime(load_config(args.config))
    try:
        asyncio.run(_serve(runtime.public_app, runtime.admin_app, runtime.config.service))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
