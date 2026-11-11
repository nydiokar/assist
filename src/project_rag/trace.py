"""Structured query diagnostics (v1 §5.2). No raw PII: query is hashed, excerpts are not logged.

Traces capture tool name, project, query hash, generation, retrieved chunk IDs, channel
contributions, result count, and timings. Used by the grounding gate: an answer without a
matching retrieval trace fails (v1 §5.2).
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from .core.fingerprint import new_query_id, query_hash  # noqa: F401  (re-export, S1 shim)
from .domain.models import QueryTrace  # noqa: F401  (re-export, S1 shim)


class _Timer:
    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}

    @contextmanager
    def span(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = round((time.perf_counter() - start) * 1000, 3)


def timer() -> _Timer:
    return _Timer()
