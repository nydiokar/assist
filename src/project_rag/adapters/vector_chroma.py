"""Chroma vector index (v1 §4.6, ADR-004). `dev-gpu` extra. PersistentClient, explicit cosine HNSW,
externally generated embeddings, project filter enforced server-side, explicit deletion of
superseded ids (never a default embedding function or metric)."""

from __future__ import annotations

import numpy as np

from ..domain.config import VectorStoreConfig
from ..domain.errors import StoreError
from ..domain.models import ChannelHit, Chunk
from .vector_payload import chunk_metadata, hit_payload


class ChromaVectorStore:
    """Production baseline. Requires the `chroma` extra. Explicit cosine, explicit embeddings."""

    def __init__(self, cfg: VectorStoreConfig) -> None:
        import chromadb
        from chromadb.config import Settings

        self._cfg = cfg
        self._client = chromadb.PersistentClient(
            path=cfg.persist_path, settings=Settings(anonymized_telemetry=False)
        )
        if cfg.distance != "cosine":
            raise ValueError("v1 §4.6 mandates cosine; config distance must be 'cosine'")
        # Explicit HNSW cosine; NO default embedding function (embeddings supplied externally).
        self._col = self._client.get_or_create_collection(
            name=cfg.collection,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def _batches(self, n: int) -> range:
        # v1 §4.6: stay within the installed client's batch limit (it is ~5.4k, below corpus size).
        return range(0, n, max(1, self._client.get_max_batch_size()))

    def add(self, generation: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        mat = np.asarray(embeddings, dtype=np.float32)
        step = max(1, self._client.get_max_batch_size())
        for i in self._batches(len(chunks)):
            part = chunks[i : i + step]
            self._col.add(
                ids=[c.chunk_id for c in part],
                embeddings=[v.tolist() for v in mat[i : i + step]],
                documents=[c.citation_text for c in part],
                metadatas=[chunk_metadata(c, generation) for c in part],
            )

    def delete_ids(self, ids: list[str]) -> None:
        step = max(1, self._client.get_max_batch_size())
        for i in range(0, len(ids), step):
            self._col.delete(ids=ids[i : i + step])

    def delete_generation(self, project_id: str, generation: str) -> None:
        self._col.delete(where={"$and": [{"project_id": project_id}, {"generation": generation}]})

    def query(
        self,
        project_id: str,
        generation: str,
        query_embedding: np.ndarray,
        limit: int,
        doc_class: str | None = None,
    ) -> list[ChannelHit]:
        clauses: list[dict] = [{"project_id": project_id}, {"generation": generation}]
        if doc_class is not None:
            clauses.append({"doc_class": doc_class})
        res = self._col.query(
            query_embeddings=[np.asarray(query_embedding, dtype=np.float32).ravel().tolist()],
            n_results=limit,
            where={"$and": clauses},
            include=["metadatas", "distances"],
        )
        if res["distances"] is None or res["metadatas"] is None:
            raise StoreError("chroma query returned no distances/metadatas")
        ids = res["ids"][0]
        dists = res["distances"][0]
        metas = res["metadatas"][0]
        return [
            ChannelHit(chunk_id=cid, score=float(d), payload=hit_payload(m))
            for cid, d, m in zip(ids, dists, metas, strict=True)
        ]

    def count(self, project_id: str, generation: str) -> int:
        return self._col.get(
            where={"$and": [{"project_id": project_id}, {"generation": generation}]}
        )["ids"].__len__()
