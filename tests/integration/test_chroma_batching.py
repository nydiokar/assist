"""Chroma writes must respect the client's max batch size (real failure: 6118 > 5461)."""

from __future__ import annotations

import pytest

pytest.importorskip("chromadb")


def test_ingest_larger_than_chroma_batch_limit(make_rag, corpus_root):
    rag = make_rag(**{"vector_store.provider": "chroma", "vector_store.collection": "t_batch"})
    rag.services.vector._client.get_max_batch_size = lambda: 3  # force many small batches
    report = rag.ingest_project("demo", corpus_root)
    assert not report.warnings or not any("REFUSED" in w for w in report.warnings)
    n = rag.get_ingestion_status("demo").n_chunks
    assert n > 3 and rag.services.vector.count("demo", report.index_generation) == n
    assert rag.search_project("demo", "GV-2025-017").results
