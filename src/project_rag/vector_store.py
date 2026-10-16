"""Vector store adapters (v1 §4.6, ADR-004, v2 §D3).

VectorStoreProtocol is the stable seam. InMemoryVectorStore (numpy cosine) backs tests and
offline validation. ChromaVectorStore is the production baseline: PersistentClient, explicit
cosine HNSW, explicit externally-generated embeddings, project filter enforced server-side,
explicit deletion of superseded IDs (never relies on a default embedding function/metric).

Metadata stored per chunk: project_id, document_id, version_id, generation, source_path,
heading, source_type, location_json (v1 §4.6 collection model).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import VectorStoreConfig
from .schema import Chunk, SourceLocation


class VectorStoreProtocol(Protocol):
    def add(self, generation: str, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    def delete_ids(self, ids: list[str]) -> None: ...

    def delete_generation(self, project_id: str, generation: str) -> None: ...

    def query(self, project_id: str, generation: str, query_embedding: np.ndarray,
              limit: int) -> list[tuple[str, float, dict]]: ...

    def count(self, project_id: str, generation: str) -> int: ...


def _payload(c: Chunk, generation: str) -> dict:
    return {
        "project_id": c.project_id,
        "document_id": c.document_id,
        "version_id": c.version_id,
        "generation": generation,
        "source_path": c.source_path,
        "heading": c.heading or "",
        "source_type": c.source_type,
        "location_json": c.location.model_dump_json(),
        "citation_text": c.citation_text,
    }


def _result_payload(meta: dict) -> dict:
    return {
        "document_id": meta["document_id"],
        "source_path": meta["source_path"],
        "heading": meta.get("heading", ""),
        "location": SourceLocation.model_validate_json(meta["location_json"]),
        "citation_text": meta["citation_text"],
    }


class InMemoryVectorStore:
    """Deterministic numpy cosine store. Test/dev only (asserted out of release).

    Persists to a JSON sidecar so the CLI's separate ingest/search processes share state
    (v2 §D3): without persistence the dense channel would be empty across invocations. When
    persist_path is empty the store stays purely in-memory (unit tests).
    """

    def __init__(self, cfg: VectorStoreConfig) -> None:
        self._cfg = cfg
        self._ids: list[str] = []
        self._vecs: list[np.ndarray] = []
        self._meta: list[dict] = []
        self._file = (Path(cfg.persist_path) / "memory_store.json"
                      if cfg.persist_path else None)
        self._load()

    def _load(self) -> None:
        if not self._file or not self._file.is_file():
            return
        data = json.loads(self._file.read_text(encoding="utf-8"))
        self._ids = data["ids"]
        self._vecs = [np.asarray(v, dtype=np.float32) for v in data["vecs"]]
        self._meta = data["meta"]

    def _save(self) -> None:
        if not self._file:
            return
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps({
            "ids": self._ids,
            "vecs": [v.tolist() for v in self._vecs],
            "meta": self._meta,
        }), encoding="utf-8")

    def add(self, generation: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        for c, v in zip(chunks, embeddings, strict=True):
            self._ids.append(c.chunk_id)
            self._vecs.append(np.asarray(v, dtype=np.float32))
            self._meta.append(_payload(c, generation))
        self._save()

    def delete_ids(self, ids: list[str]) -> None:
        keep = [i for i, cid in enumerate(self._ids) if cid not in set(ids)]
        self._ids = [self._ids[i] for i in keep]
        self._vecs = [self._vecs[i] for i in keep]
        self._meta = [self._meta[i] for i in keep]
        self._save()

    def delete_generation(self, project_id: str, generation: str) -> None:
        drop = [self._ids[i] for i, m in enumerate(self._meta)
                if m["project_id"] == project_id and m["generation"] == generation]
        self.delete_ids(drop)

    def query(self, project_id: str, generation: str, query_embedding: np.ndarray,
              limit: int) -> list[tuple[str, float, dict]]:
        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        qn = np.linalg.norm(q) or 1.0
        scored: list[tuple[str, float, dict]] = []
        for cid, v, m in zip(self._ids, self._vecs, self._meta, strict=True):
            if m["project_id"] != project_id or m["generation"] != generation:
                continue  # project filter enforced BEFORE ranking (v1 §4.6)
            vn = np.linalg.norm(v) or 1.0
            cos = float(np.dot(q, v) / (qn * vn))
            scored.append((cid, 1.0 - cos, _result_payload(m)))  # cosine distance
        scored.sort(key=lambda t: t[1])
        return scored[:limit]

    def count(self, project_id: str, generation: str) -> int:
        return sum(1 for m in self._meta
                   if m["project_id"] == project_id and m["generation"] == generation)


class ChromaVectorStore:
    """Production baseline. Requires the `chroma` extra. Explicit cosine, explicit embeddings."""

    def __init__(self, cfg: VectorStoreConfig) -> None:
        import chromadb  # noqa: PLC0415
        from chromadb.config import Settings  # noqa: PLC0415

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

    def add(self, generation: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=[v.tolist() for v in np.asarray(embeddings, dtype=np.float32)],
            documents=[c.citation_text for c in chunks],
            metadatas=[_payload(c, generation) for c in chunks],
        )

    def delete_ids(self, ids: list[str]) -> None:
        if ids:
            self._col.delete(ids=ids)

    def delete_generation(self, project_id: str, generation: str) -> None:
        self._col.delete(where={"$and": [{"project_id": project_id}, {"generation": generation}]})

    def query(self, project_id: str, generation: str, query_embedding: np.ndarray,
              limit: int) -> list[tuple[str, float, dict]]:
        res = self._col.query(
            query_embeddings=[np.asarray(query_embedding, dtype=np.float32).ravel().tolist()],
            n_results=limit,
            where={"$and": [{"project_id": project_id}, {"generation": generation}]},
            include=["metadatas", "distances"],
        )
        ids = res["ids"][0]
        dists = res["distances"][0]
        metas = res["metadatas"][0]
        return [(cid, float(d), _result_payload(m))
                for cid, d, m in zip(ids, dists, metas, strict=True)]

    def count(self, project_id: str, generation: str) -> int:
        return self._col.get(
            where={"$and": [{"project_id": project_id}, {"generation": generation}]}
        )["ids"].__len__()


def build_vector_store(cfg: VectorStoreConfig) -> VectorStoreProtocol:
    if cfg.provider == "memory":
        return InMemoryVectorStore(cfg)
    if cfg.provider == "chroma":
        return ChromaVectorStore(cfg)
    raise ValueError(f"unknown vector_store.provider: {cfg.provider}")
