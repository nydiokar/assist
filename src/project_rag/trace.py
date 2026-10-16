"""Structured query diagnostics (v1 §5.2). No raw PII: query is hashed, excerpts are not logged.

Traces capture tool name, project, query hash, generation, retrieved chunk IDs, channel
contributions, result count, and timings. Used by the grounding gate: an answer without a
matching retrieval trace fails (v1 §5.2).
"""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager

from pydantic import BaseModel, Field


class QueryTrace(BaseModel):
    query_id: str
    tool: str
    project_id: str
    query_hash: str
    index_generation: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    channels: dict[str, int] = Field(default_factory=dict)  # channel -> candidate count
    result_count: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def new_query_id(project_id: str, query: str, nonce: str) -> str:
    raw = f"{project_id}|{query}|{nonce}"
    return "q_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


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
