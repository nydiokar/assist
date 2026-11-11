"""Identity reranker (tests): keeps the incoming order by scoring it strictly descending."""

from __future__ import annotations


class IdentityReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        n = len(passages)
        return [float(n - i) for i in range(n)]
