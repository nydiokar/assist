"""Service runtime (S6): the composition of the long-lived assist service.

Builds the read-side services on the single store thread, the single-writer ingest worker (its
own services and connection, on its own thread), and the public/admin ASGI apps. Like
`wiring`, this module may know adapters; nothing below it does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .adapters.feedback_sqlite import SqliteFeedbackSink
from .adapters.telemetry_prometheus import PrometheusTelemetry
from .domain.config import Config
from .domain.models import (
    FeedbackSignal,
    IngestReport,
    ProjectIdentity,
    SearchAllResponse,
    SearchResponse,
    SourceLocation,
    SourceResponse,
    StatusReport,
)
from .interfaces.http.app import build_admin_app, build_public_app
from .interfaces.http.executor import StoreExecutor
from .interfaces.http.ingest_worker import IngestWorker
from .wiring import Services, build

TOKEN_ENV = "ASSIST_MCP_TOKEN"


class ServiceApiImpl:
    """Read-side operations for the HTTP tools. Every method runs on the store thread."""

    def __init__(self, services: Services, telemetry: PrometheusTelemetry, config: Config) -> None:
        self._s = services
        self._telemetry = telemetry
        self._cfg = config
        self._links: dict[str, str] = {}
        self._folders: dict[str, str] = {}
        self._ids: frozenset[str] = frozenset()

    # -- registry / roots --
    def sync_registry(self) -> None:
        root = self._cfg.service.corpus_root
        for rec in self._s.registry.sync(root):
            self._s.search.register_project_root(rec.identity.project_id, rec.root_path)
            self._folders[rec.identity.project_id] = rec.folder
        self.projects()
        manifest = self._s.registry.read_manifest(root)
        self._links = {f.path: f.web_link for f in manifest.files} if manifest else {}

    def projects(self) -> list[ProjectIdentity]:
        projects = self._s.catalog.list_projects()
        self._ids = frozenset(p.project_id for p in projects)
        return projects

    def cached_project_ids(self) -> frozenset[str]:
        return self._ids

    def _ensure_root(self, project_id: str) -> None:
        if project_id in self._folders:
            return
        for p in self._s.catalog.list_projects():
            if p.project_id == project_id:
                root = self._s.registry_corpus.resolve(
                    self._cfg.service.corpus_root, p.display_name
                )
                self._s.search.register_project_root(project_id, root)
                self._folders[project_id] = p.display_name

    # -- tools --
    def search(
        self, project_id: str, query: str, top_k: int, doc_class: str | None
    ) -> SearchResponse:
        self._ensure_root(project_id)
        filters = {"doc_class": doc_class} if doc_class else None
        return self._s.search.search_project(project_id, query, top_k=top_k, filters=filters)

    def search_all(self, query: str, project_ids: list[str], top_k: int) -> SearchAllResponse:
        for pid in project_ids:
            self._ensure_root(pid)
        return self._s.search.search_all(query, project_ids, top_k=top_k)

    def open_source(
        self, project_id: str, document_id: str, source_path: str, location: SourceLocation | None
    ) -> SourceResponse:
        self._ensure_root(project_id)
        return self._s.search.open_source(
            project_id, document_id, source_path=source_path, location=location
        )

    def status(self, project_id: str) -> StatusReport:
        return self._s.ingest.status(project_id)

    def web_link(self, project_id: str, source_path: str) -> str | None:
        folder = self._folders.get(project_id)
        link = self._links.get(f"{folder}/{source_path}") if folder else None
        return link or None

    def record_feedback(self, signal: FeedbackSignal) -> None:
        if self._s.feedback is not None:
            self._s.feedback.record(signal)
        self._telemetry.count("assist_feedback_total", {"kind": signal.kind})

    def now(self) -> datetime:
        return self._s.clock.now()

    # -- health / metrics --
    def health(self) -> dict[str, str]:
        out: dict[str, str] = {}
        out["embed"] = _probe(self._s.embedder)
        out["rerank"] = _probe(self._s.reranker) if self._s.reranker is not None else "ok"
        try:
            self._s.catalog.active_generations()
            out["store"] = "ok"
        except Exception as e:  # noqa: BLE001 - health probe boundary: report, never raise
            out["store"] = f"error: {type(e).__name__}"
        return out

    def refresh_metrics(self) -> None:
        now = self._s.clock.now()
        for p in self._s.catalog.list_projects():
            st = self._s.ingest.status(p.project_id)
            for state, n in st.states.items():
                self._telemetry.gauge("assist_docs", n, {"project": p.project_id, "state": state})
            runs = [r for r in self._s.catalog.runs(p.project_id, 20) if r.published]
            if runs:
                last = runs[0]
                self._telemetry.gauge(
                    "assist_coverage_ratio", last.coverage_pct / 100.0, {"project": p.project_id}
                )
                age = (now - datetime.fromisoformat(last.finished_at)).total_seconds()
                self._telemetry.gauge("assist_index_age_seconds", age, {"project": p.project_id})

    def metrics_text(self) -> bytes:
        return self._telemetry.exposition()


def _probe(provider: object) -> str:
    healthy = getattr(provider, "healthy", None)
    if healthy is None:
        return "ok"  # in-process provider (hash/torch): nothing to reach
    try:
        return "ok" if healthy() else "unreachable"
    except Exception as e:  # noqa: BLE001 - health probe boundary: report, never raise
        return f"error: {type(e).__name__}"


MAINT_KEY_ENV = "ANTHROPIC_API_KEY_MAINT"  # D-7: owner key for tier-S only


class WriterImpl:
    """Writer-side services for the ingest worker (own store connection, built on its thread).
    After a project's FIRST published ingest it runs tier-S smoke (spec §6.8.3) and stores the
    result + operator message next to the run reports."""

    def __init__(self, config: Config, question_writer: object | None = None) -> None:
        self._s = build(config, collect_garbage=False)
        self._writer = question_writer
        self.smoke_runs: list[object] = []

    def ingest_corpus(self, corpus_root: str, project_id: str | None) -> list[IngestReport]:
        reports: list[IngestReport] = []
        for rec in self._s.registry.sync(corpus_root):
            if project_id is None or rec.identity.project_id == project_id:
                report = self._s.ingest.ingest(rec.identity.project_id, rec.root_path)
                reports.append(report)
                run = report.run
                if run is not None and run.published and run.prior_generation is None:
                    self._smoke(rec.identity.project_id, rec.identity.display_name, rec.root_path)
        return reports

    def _smoke(self, project_id: str, display_name: str, root: str) -> None:
        from pathlib import Path

        from .services.smoke import TierSService

        out_dir = Path(self._s.config.lexical_store.path).resolve().parent / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        if self._writer is None:
            (out_dir / f"smoke_{project_id}.json").write_text(
                '{"skipped": "no tier-S writer configured (ANTHROPIC_API_KEY_MAINT unset, D-7)"}',
                encoding="utf-8",
            )
            return
        self._s.search.register_project_root(project_id, root)
        svc = TierSService(
            self._writer,  # type: ignore[arg-type]
            self._s.lexical,
            self._s.catalog,
            lambda pid, q: self._s.search.search_project(pid, q, top_k=5),
            self._s.ingest.status,
        )
        result = svc.run(project_id, display_name)
        self.smoke_runs.append(result)
        (out_dir / f"smoke_{project_id}.json").write_text(
            result.model_dump_json(indent=1), encoding="utf-8"
        )

    def close(self) -> None:
        self._s.close()


@dataclass
class Runtime:
    config: Config
    services: Services
    api: ServiceApiImpl
    executor: StoreExecutor
    worker: IngestWorker
    telemetry: PrometheusTelemetry
    public_app: Any
    admin_app: Any

    def close(self) -> None:
        self.executor.submit_sync(self.services.close)
        self.executor.shutdown()


def build_runtime(config: Config, *, token: str | None = None) -> Runtime:
    """Start-up order: refuse unsafe configs, build services ON the store thread (GC runs there),
    sync the registry, then build the apps."""
    problems = config.assert_release_safe()
    if config.service.verify_models:
        from .adapters.model_check import gguf_problems

        problems += gguf_problems(config, config.service.models_dir)
    if problems and os.environ.get("ASSIST_ALLOW_DEV_PROFILE") != "1":
        raise SystemExit("refusing to start: " + "; ".join(problems))
    mcp_token = token if token is not None else os.environ.get(TOKEN_ENV, "")
    if not mcp_token:
        raise SystemExit(f"refusing to start: {TOKEN_ENV} is not set")
    telemetry = PrometheusTelemetry()
    executor = StoreExecutor(max_waiting=config.service.max_waiting)
    services = executor.submit_sync(lambda: build(config, telemetry=telemetry, feedback=True))
    api = ServiceApiImpl(services, telemetry, config)
    executor.submit_sync(api.sync_registry)
    maint_key = os.environ.get(MAINT_KEY_ENV, "")

    def open_writer() -> WriterImpl:
        qw = None
        if maint_key:
            import anthropic

            from .adapters.writer_anthropic import AnthropicQuestionWriter

            qw = AnthropicQuestionWriter(client=anthropic.Anthropic(api_key=maint_key))
        return WriterImpl(config, qw)

    worker = IngestWorker(open_writer, config.service.corpus_root)
    return Runtime(
        config=config,
        services=services,
        api=api,
        executor=executor,
        worker=worker,
        telemetry=telemetry,
        public_app=build_public_app(api, executor, config.service, mcp_token),
        admin_app=build_admin_app(api, executor, worker, config.service),
    )


__all__ = ["Runtime", "SqliteFeedbackSink", "build_runtime"]
