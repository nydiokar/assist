"""OpenAI-compatible embedder for llama-server (spec §6.1, A1/A3/A4).

- `POST /v1/embeddings`, batched (`max_batch`), explicit timeouts, bounded retries.
- Queries get the exact `qwen3_query_v1` instruct prompt prepended CLIENT-side; passages none.
- Vectors are validated (dim, finite) and L2-normalised client-side; a wrong dim or a NaN is a
  `ProviderContractViolation` (never retried, never stored).
- Token counts come from `POST /tokenize` of the SERVED model (`add_special` on, matching the HF
  tokenizer's `add_special_tokens=True`), cached by text hash.
"""

from __future__ import annotations

import hashlib
import math

import httpx
import numpy as np
import numpy.typing as npt

from ..core.vectors import l2_normalize
from ..domain.config import EmbeddingConfig
from ..domain.errors import ProviderContractViolation
from ..domain.models import EmbeddingProfile
from .http_provider import ProviderClient


class OpenAICompatEmbedder:
    def __init__(
        self, cfg: EmbeddingConfig, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._o = cfg.openai_compat
        self._profile = cfg.profile()
        self._http = ProviderClient(
            self._o.base_url,
            timeout_s=self._o.timeout_s,
            connect_timeout_s=self._o.connect_timeout_s,
            retry=self._o.retry,
            api_key=self._o.api_key,
            transport=transport,
        )
        self._token_cache: dict[str, int] = {}

    @property
    def dim(self) -> int:
        return self._o.dim

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    @property
    def max_input_tokens(self) -> int:
        return self._o.max_input_tokens

    def token_count(self, text: str) -> int:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._token_cache.get(key)
        if cached is not None:
            return cached
        body = self._http.post_json(
            "/tokenize", {"content": text + self._o.append_text, "add_special": True}
        )
        tokens = body.get("tokens") if isinstance(body, dict) else None
        if not isinstance(tokens, list):
            raise ProviderContractViolation("/tokenize: response has no 'tokens' list")
        n = len(tokens)  # pyright: ignore[reportUnknownArgumentType]
        self._token_cache[key] = n
        return n

    def _embed(self, inputs: list[str]) -> npt.NDArray[np.float32]:
        out = np.zeros((len(inputs), self._o.dim), dtype=np.float32)
        for start in range(0, len(inputs), max(1, self._o.max_batch)):
            part = [t + self._o.append_text for t in inputs[start : start + self._o.max_batch]]
            body = self._http.post_json(
                "/v1/embeddings",
                {"model": self._o.model, "input": part, "encoding_format": "float"},
            )
            vectors = _parse_embeddings(body, len(part), self._o.dim)
            out[start : start + len(part)] = vectors
        return l2_normalize(out) if self._o.normalize else out

    def embed_passages(self, texts: list[str]) -> npt.NDArray[np.float32]:
        return self._embed(list(texts))

    def embed_queries(self, texts: list[str]) -> npt.NDArray[np.float32]:
        return self._embed([self._o.query_prompt + t for t in texts])

    def healthy(self) -> bool:
        return self._http.get_ok("/health")

    def close(self) -> None:
        self._http.close()


def _parse_embeddings(body: object, expected: int, dim: int) -> npt.NDArray[np.float32]:
    """Decode an OpenAI-style embeddings body into (expected, dim) float32, in input order."""
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or len(data) != expected:  # pyright: ignore[reportUnknownArgumentType]
        raise ProviderContractViolation(f"embeddings: expected {expected} items in 'data'")
    rows: list[tuple[int, list[float]]] = []
    for i, item in enumerate(data):  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(item, dict):
            raise ProviderContractViolation("embeddings: item is not an object")
        vec = item.get("embedding")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        idx = item.get("index", i)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if not isinstance(vec, list) or not isinstance(idx, int):
            raise ProviderContractViolation("embeddings: item lacks embedding/index")
        if len(vec) != dim:  # pyright: ignore[reportUnknownArgumentType]
            raise ProviderContractViolation(f"embeddings: dim {len(vec)} != profile dim {dim}")  # pyright: ignore[reportUnknownArgumentType]
        floats = [float(x) for x in vec]  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
        if not all(math.isfinite(x) for x in floats):
            raise ProviderContractViolation("embeddings: non-finite value")
        rows.append((idx, floats))
    rows.sort(key=lambda r: r[0])
    if [r[0] for r in rows] != list(range(expected)):
        raise ProviderContractViolation("embeddings: indexes are not 0..n-1")
    return np.asarray([r[1] for r in rows], dtype=np.float32)
