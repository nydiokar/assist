"""VectorIndex port contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from project_rag.adapters.vector_memory import InMemoryVectorStore
from project_rag.domain.config import VectorStoreConfig
from project_rag.ports.vector import VectorIndex
from tests.contracts.conftest import make_chunk

DIM = 8


def _memory(tmp_path: Path) -> VectorIndex:
    return InMemoryVectorStore(VectorStoreConfig(provider="memory", persist_path=""))


def _memory_persisted(tmp_path: Path) -> VectorIndex:
    return InMemoryVectorStore(
        VectorStoreConfig(provider="memory", persist_path=str(tmp_path / "mem"))
    )


def _chroma(tmp_path: Path) -> VectorIndex:
    pytest.importorskip("chromadb")
    from project_rag.adapters.vector_chroma import ChromaVectorStore

    return ChromaVectorStore(
        VectorStoreConfig(
            provider="chroma", persist_path=str(tmp_path / "chroma"), collection="contract"
        )
    )


ADAPTERS: list[object] = [
    pytest.param(_memory, id="memory"),
    pytest.param(_memory_persisted, id="memory_persisted"),
    pytest.param(_chroma, id="chroma"),
]


@pytest.fixture(params=ADAPTERS)
def index(request: pytest.FixtureRequest, tmp_path: Path) -> VectorIndex:
    factory: Callable[[Path], VectorIndex] = request.param
    return factory(tmp_path)


def _unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def test_nearest_first_and_scoped(index: VectorIndex) -> None:
    index.add(
        "g1", [make_chunk("c0", "zero"), make_chunk("c1", "one")], np.stack([_unit(0), _unit(1)])
    )
    index.add("g2", [make_chunk("c9", "other gen")], np.stack([_unit(0)]))
    index.add("g1", [make_chunk("cx", "other project", project_id="p2")], np.stack([_unit(0)]))
    q = (_unit(0) * 0.9 + _unit(1) * 0.1).astype(np.float32)
    hits = index.query("p1", "g1", q, 5)
    assert [h.chunk_id for h in hits] == ["c0", "c1"]
    assert hits[0].score <= hits[1].score  # cosine distance ascending
    assert hits[0].payload.citation_text == "zero"
    assert index.count("p1", "g1") == 2


def test_doc_class_filter_and_limit(index: VectorIndex) -> None:
    chunks = [make_chunk("c0", "a", doc_class="reference"), make_chunk("c1", "b")]
    index.add("g1", chunks, np.stack([_unit(0), _unit(1)]))
    assert [h.chunk_id for h in index.query("p1", "g1", _unit(1), 5, "reference")] == ["c0"]
    assert len(index.query("p1", "g1", _unit(1), 1)) == 1


def test_delete_ids_and_generation(index: VectorIndex) -> None:
    index.add("g1", [make_chunk("c0", "a"), make_chunk("c1", "b")], np.stack([_unit(0), _unit(1)]))
    index.delete_ids(["c0"])
    assert [h.chunk_id for h in index.query("p1", "g1", _unit(0), 5)] == ["c1"]
    index.delete_generation("p1", "g1")
    assert index.count("p1", "g1") == 0
