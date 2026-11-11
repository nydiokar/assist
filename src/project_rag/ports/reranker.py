"""Reranker port: a content-aware relevance score per (query, passage); higher is better."""

from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        """One score per passage, same order; [] for no passages."""
        ...
