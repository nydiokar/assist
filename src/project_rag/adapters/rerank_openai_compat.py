"""OpenAI/Jina-style reranker over llama-server `--reranking` (`POST /v1/rerank`; spec §6.1, A2).

Returns one score per passage in input order. `score_transform="sigmoid"` maps raw logits to the
0-1 scale the torch CrossEncoder produced (the abstention threshold is on that scale).
"""

from __future__ import annotations

import math

import httpx

from ..domain.config import RerankConfig
from ..domain.errors import ProviderContractViolation
from .http_provider import ProviderClient


class OpenAICompatReranker:
    def __init__(self, cfg: RerankConfig, *, transport: httpx.BaseTransport | None = None) -> None:
        self._o = cfg.openai_compat
        self._http = ProviderClient(
            self._o.base_url,
            timeout_s=self._o.timeout_s,
            connect_timeout_s=self._o.connect_timeout_s,
            retry=self._o.retry,
            transport=transport,
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        body = self._http.post_json(
            "/v1/rerank",
            {
                "model": self._o.model,
                "query": query,
                "documents": passages,
                "top_n": len(passages),
            },
        )
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            raise ProviderContractViolation("rerank: response has no 'results' list")
        scores: dict[int, float] = {}
        for item in results:  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(item, dict):
                raise ProviderContractViolation("rerank: result is not an object")
            idx = item.get("index")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            val = item.get("relevance_score")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            if not isinstance(idx, int) or not isinstance(val, int | float):
                raise ProviderContractViolation("rerank: result lacks index/relevance_score")
            if not math.isfinite(float(val)):
                raise ProviderContractViolation("rerank: non-finite score")
            scores[idx] = float(val)
        if sorted(scores) != list(range(len(passages))):
            raise ProviderContractViolation("rerank: results do not cover every passage")
        raw = [scores[i] for i in range(len(passages))]
        if self._o.score_transform == "sigmoid":
            return [1.0 / (1.0 + math.exp(-x)) for x in raw]
        return raw

    def healthy(self) -> bool:
        return self._http.get_ok("/health")

    def close(self) -> None:
        self._http.close()
