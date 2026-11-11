"""Prometheus telemetry (spec §6.10). Never raises: a metrics bug must not fail a request (R6)."""

from __future__ import annotations

from collections.abc import Mapping

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


class PrometheusTelemetry:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self._stage = Histogram(
            "assist_stage_latency_seconds",
            "Search stage latency",
            ["stage"],
            registry=r,
            buckets=_BUCKETS,
        )
        self._counters: dict[str, Counter] = {
            "assist_search_total": Counter(
                "assist_search_total", "Searches by outcome", ["outcome"], registry=r
            ),
            "assist_degraded_total": Counter(
                "assist_degraded_total",
                "Degraded responses by component",
                ["component"],
                registry=r,
            ),
            "assist_feedback_total": Counter(
                "assist_feedback_total", "Feedback signals by kind", ["kind"], registry=r
            ),
            "assist_rejected_total": Counter(
                "assist_rejected_total", "Requests refused by the bounded queue", [], registry=r
            ),
        }
        self._gauges: dict[str, Gauge] = {
            "assist_index_age_seconds": Gauge(
                "assist_index_age_seconds",
                "Age of the active generation",
                ["project"],
                registry=r,
            ),
            "assist_coverage_ratio": Gauge(
                "assist_coverage_ratio", "READY / in-scope documents", ["project"], registry=r
            ),
            "assist_docs": Gauge(
                "assist_docs", "Documents by state", ["project", "state"], registry=r
            ),
            "sync_last_success_timestamp": Gauge(
                "sync_last_success_timestamp",
                "Last successful Drive sync (unix s)",
                [],
                registry=r,
            ),
            "sync_changed_files": Gauge(
                "sync_changed_files", "Files changed by the last sync", [], registry=r
            ),
        }

    def observe(self, stage: str, seconds: float) -> None:
        try:
            self._stage.labels(stage=stage).observe(max(0.0, seconds))
        except Exception:  # noqa: BLE001 - telemetry must never raise (port contract)
            return

    def count(self, name: str, labels: Mapping[str, str] | None = None) -> None:
        try:
            c = self._counters.get(name)
            if c is None:
                return
            (c.labels(**dict(labels)) if labels else c).inc()
        except Exception:  # noqa: BLE001 - telemetry must never raise (port contract)
            return

    def gauge(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        try:
            g = self._gauges.get(name)
            if g is None:
                return
            (g.labels(**dict(labels)) if labels else g).set(value)
        except Exception:  # noqa: BLE001 - telemetry must never raise (port contract)
            return

    def exposition(self) -> bytes:
        return generate_latest(self.registry)
