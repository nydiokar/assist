"""The assist ASGI apps (spec §6.3, S6).

- public app (agent network): `/mcp` (MCP Streamable HTTP, static bearer token), `/healthz`
  (embed, rerank, store), `/metrics` (Prometheus text).
- admin app (knowledge network only, its own listener): `POST /admin/ingest[?project=]`,
  `GET /admin/status`.
Bounded queue -> 503 + Retry-After. Request bodies over `max_body_bytes` -> 413.
"""

from __future__ import annotations

import contextlib
import hmac
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from ...domain.config import ServiceConfig
from ..mcp.http_tools import ServiceApi, build_http_server
from .executor import DeadlineExceeded, Overloaded, StoreExecutor
from .ingest_worker import IngestWorker

Scope = MutableMapping[str, Any]  # ASGI edge types (Any only here, per R9)
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]


BUSY_MESSAGE = "assist is busy ({detail}); retry in a few seconds"


class ServiceOps(ServiceApi, Protocol):
    def health(self) -> dict[str, str]: ...

    def refresh_metrics(self) -> None: ...

    def metrics_text(self) -> bytes: ...


class _Guard:
    """Bearer token on /mcp and a request-body cap, as plain ASGI middleware."""

    def __init__(
        self,
        app: Any,
        token: str,
        max_body: int,
        executor: StoreExecutor | None = None,
        retry_after_s: int = 2,
    ) -> None:
        self._app = app
        self._token = token.encode()
        self._max_body = max_body
        self._executor = executor
        self._retry_after = str(retry_after_s)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = {k.decode().lower(): v for k, v in scope.get("headers", [])}
            length = int(headers.get("content-length", b"0") or b"0")
            if length > self._max_body:
                await JSONResponse({"error": "request body too large"}, 413)(scope, receive, send)
                return
            if str(scope.get("path", "")).startswith("/mcp"):
                given = headers.get("authorization", b"")
                if not self._token or not hmac.compare_digest(given, b"Bearer " + self._token):
                    await JSONResponse({"error": "unauthorized"}, 401)(scope, receive, send)
                    return
                if self._executor is not None and self._executor.saturated:
                    # A JSON-RPC error body: MCP clients surface it verbatim at a non-2xx status,
                    # so the agent reads "busy, retry" instead of a bare transport failure.
                    await JSONResponse(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {
                                "code": -32000,
                                "message": BUSY_MESSAGE.format(detail="store queue full"),
                                "data": {"retry_after_s": float(self._retry_after)},
                            },
                        },
                        503,
                        headers={"Retry-After": self._retry_after},
                    )(scope, receive, send)
                    return
        await self._app(scope, receive, send)


def build_public_app(
    ops: ServiceOps, executor: StoreExecutor, cfg: ServiceConfig, token: str
) -> Any:
    mcp = build_http_server(ops, executor, cfg)
    from mcp.server.transport_security import TransportSecuritySettings

    security = TransportSecuritySettings(
        enable_dns_rebinding_protection="*" not in cfg.allowed_hosts,
        allowed_hosts=[h for h in cfg.allowed_hosts if h != "*"],
    )
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        max_request_body_size=cfg.max_body_bytes,
        transport_security=security,
        host=cfg.mcp_host,
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async with mcp.session_manager.run():
            yield

    app = FastAPI(title="assist", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.get("/healthz")
    async def healthz() -> Response:
        try:
            components = await executor.run(ops.health, cfg.status_deadline_s)
        except (Overloaded, DeadlineExceeded) as e:
            return JSONResponse({"status": "busy", "detail": str(e)}, 503)
        ok = all(v == "ok" for v in components.values())
        store_ok = components.get("store") == "ok"
        status = "ok" if ok else ("degraded" if store_ok else "down")
        return JSONResponse({"status": status, "components": components}, 200 if store_ok else 503)

    @app.get("/metrics")
    async def metrics() -> Response:
        with contextlib.suppress(Overloaded, DeadlineExceeded):
            await executor.run(ops.refresh_metrics, cfg.status_deadline_s)
        return PlainTextResponse(ops.metrics_text(), media_type="text/plain; version=0.0.4")

    @app.exception_handler(Overloaded)
    async def overloaded(_request: Request, exc: Overloaded) -> Response:
        return JSONResponse(
            {"error": "busy", "detail": str(exc)},
            503,
            headers={"Retry-After": str(cfg.retry_after_s)},
        )

    app.mount("/", mcp_app)
    return _Guard(app, token, cfg.max_body_bytes, executor, cfg.retry_after_s)


def build_admin_app(
    ops: ServiceOps, executor: StoreExecutor, worker: IngestWorker, cfg: ServiceConfig
) -> Any:
    app = FastAPI(title="assist-admin", docs_url=None, redoc_url=None)

    @app.post("/admin/ingest")
    async def ingest(project: str | None = None) -> Response:
        state = worker.trigger(project)
        return JSONResponse({"ingest": state, "project": project}, 202)

    @app.get("/admin/status")
    async def status() -> Response:
        projects = await executor.run(ops.projects, cfg.status_deadline_s)
        rows = []
        for p in projects:
            st = await executor.run(lambda pid=p.project_id: ops.status(pid), cfg.status_deadline_s)
            rows.append(st.model_dump())
        return JSONResponse(
            {
                "ingest_running": worker.running,
                "ingest_runs": worker.runs,
                "last_error": worker.last_error,
                "last_reports": [
                    r.run.model_dump() if r.run else None for r in worker.last_reports
                ],
                "projects": rows,
                "store_pending": executor.pending,
            }
        )

    return _Guard(app, "", cfg.max_body_bytes)
