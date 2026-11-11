"""Cross-encoder reranker (spec v1 §7). Reorders fused candidates by a content-aware
query×chunk relevance score, fixing the mis-ranking the WP3 k-sweep localized.

Why this stage exists: RRF fusion is rank-only — it scores a chunk by its position in each
channel, never by whether the chunk text actually answers the query. So N sibling chunks of the
right document can crowd the top-5 and push the one chunk that holds the answer to rank 6-8
(measured: yp28/yp33/yp35). A cross-encoder reads (query, chunk_text) jointly and scores
relevance directly, which is the standard third stage of a production RAG pipeline.

Lazy, offline, fail-closed on a missing model. No network beyond the local HF cache (v1 §8).
"""

from __future__ import annotations

from .config import RerankConfig


class CrossEncoderReranker:
    def __init__(self, cfg: RerankConfig) -> None:
        self._cfg = cfg
        self._model = None  # lazy

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            self._cfg.model,
            revision=self._cfg.revision,
            device=self._cfg.device,
            max_length=self._cfg.max_length,
            trust_remote_code=False,
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance score per (query, passage); higher is more relevant."""
        if not passages:
            return []
        self._ensure()
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs, batch_size=self._cfg.batch_size)  # type: ignore[union-attr]
        return [float(x) for x in scores]
