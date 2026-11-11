"""Query pipeline: dense + lexical channels, RRF fusion, file-diversity dedup (v1 §4.7/§5.1).

No LLM. Project scope is enforced inside each store before fusion. Deterministic.
"""

from __future__ import annotations

import numpy as np

from .config import RetrievalConfig
from .core.fusion import rrf as _rrf  # noqa: F401  (re-export for scripts, S1 shim)
from .core.query_expansion import expand_query
from .core.ranking import apply_rerank_scores, fuse_and_rank, truncate_excerpts
from .domain.models import ChannelHit, RetrievedResult
from .embeddings import EmbedderProtocol
from .lexical_store import LexicalStore
from .vector_store import VectorStoreProtocol


class Retriever:
    def __init__(
        self,
        embedder: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
        lexical_store: LexicalStore,
        cfg: RetrievalConfig,
    ) -> None:
        self._embedder = embedder
        self._vs = vector_store
        self._lx = lexical_store
        self._cfg = cfg
        self._reranker = None
        if cfg.rerank.enabled:
            from .rerank import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(cfg.rerank)

    def retrieve(
        self,
        project_id: str,
        generation: str,
        query: str,
        excerpt_limit: int,
        *,
        doc_class: str | None = None,
        channels: tuple[str, ...] = ("dense", "lexical"),
    ) -> tuple[list[RetrievedResult], dict[str, int]]:  # fixes S0 pyright finding
        dense: list[ChannelHit] = []
        lexical: list[ChannelHit] = []
        # Cross-language query expansion (F11/F14): append counterpart-language domain anchors so
        # a short BG table row is reachable from an EN query. Additive; same expanded text feeds
        # both channels. Original query is unchanged upstream (trace hashes the original).
        search_query = query
        if self._cfg.query_expansion.enabled:
            search_query = expand_query(
                query, corpus_script=self._cfg.query_expansion.corpus_script
            )
        if "dense" in channels:
            qvec = self._embedder.embed_queries([search_query])[0]
            dense = self._vs.query(
                project_id,
                generation,
                np.asarray(qvec),
                self._cfg.candidates_per_channel,
                doc_class,
            )
        if "lexical" in channels:
            lexical = self._lx.search(
                project_id, generation, search_query, self._cfg.candidates_per_channel, doc_class
            )
        if self._reranker is not None:
            # Fuse a wider window, rerank it by content, then cut to top_k.
            pool = fuse_and_rank(dense, lexical, self._cfg, limit=self._cfg.rerank.top_n)
            if pool:
                head = pool[: self._cfg.rerank.top_n]
                scores = self._reranker.score(query, [r.excerpt or "" for r in head])
                pool = apply_rerank_scores(pool, scores, self._cfg.rerank.top_n)
            results = pool[: self._cfg.top_k]
        else:
            results = fuse_and_rank(dense, lexical, self._cfg)
        results = truncate_excerpts(results, excerpt_limit)
        counts = {"dense": len(dense), "lexical": len(lexical)}
        return results, counts
