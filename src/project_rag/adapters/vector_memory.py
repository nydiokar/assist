"""In-memory numpy-cosine vector index (v1 §4.6, v2 §D3). Test/dev only (asserted out of release).

Metadata stored per chunk: project_id, document_id, version_id, generation, source_path,
heading, source_type, location_json (v1 §4.6 collection model).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..domain.config import VectorStoreConfig
from ..domain.models import ChannelHit, Chunk
from .vector_payload import chunk_metadata, hit_payload


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
        self._file = Path(cfg.persist_path) / "memory_store.json" if cfg.persist_path else None
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
        self._file.write_text(
            json.dumps(
                {
                    "ids": self._ids,
                    "vecs": [v.tolist() for v in self._vecs],
                    "meta": self._meta,
                }
            ),
            encoding="utf-8",
        )

    def add(self, generation: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        for c, v in zip(chunks, embeddings, strict=True):
            self._ids.append(c.chunk_id)
            self._vecs.append(np.asarray(v, dtype=np.float32))
            self._meta.append(chunk_metadata(c, generation))
        self._save()

    def delete_ids(self, ids: list[str]) -> None:
        keep = [i for i, cid in enumerate(self._ids) if cid not in set(ids)]
        self._ids = [self._ids[i] for i in keep]
        self._vecs = [self._vecs[i] for i in keep]
        self._meta = [self._meta[i] for i in keep]
        self._save()

    def delete_generation(self, project_id: str, generation: str) -> None:
        drop = [
            self._ids[i]
            for i, m in enumerate(self._meta)
            if m["project_id"] == project_id and m["generation"] == generation
        ]
        self.delete_ids(drop)

    def query(
        self,
        project_id: str,
        generation: str,
        query_embedding: np.ndarray,
        limit: int,
        doc_class: str | None = None,
    ) -> list[ChannelHit]:
        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        qn = np.linalg.norm(q) or 1.0
        scored: list[ChannelHit] = []
        for cid, v, m in zip(self._ids, self._vecs, self._meta, strict=True):
            if m["project_id"] != project_id or m["generation"] != generation:
                continue  # project filter enforced BEFORE ranking (v1 §4.6)
            if doc_class is not None and m.get("doc_class", "project_record") != doc_class:
                continue
            vn = np.linalg.norm(v) or 1.0
            cos = float(np.dot(q, v) / (qn * vn))
            # cosine distance
            scored.append(ChannelHit(chunk_id=cid, score=1.0 - cos, payload=hit_payload(m)))
        scored.sort(key=lambda t: t.score)
        return scored[:limit]

    def count(self, project_id: str, generation: str) -> int:
        return sum(
            1 for m in self._meta if m["project_id"] == project_id and m["generation"] == generation
        )
