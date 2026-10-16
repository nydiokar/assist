"""Embedding cache (A01 §C stage [3], ledger F13): unchanged chunks are embedded once."""

from __future__ import annotations

import shutil


def test_only_changed_chunks_are_reembedded(make_rag, corpus_root, tmp_path):
    root = tmp_path / "corpus"
    shutil.copytree(corpus_root, root)
    rag = make_rag()
    emb = rag._indexer._embedder
    seen: list[int] = []
    orig = emb.embed_passages
    emb.embed_passages = lambda texts: (seen.append(len(texts)), orig(texts))[1]

    rag.ingest_project("demo", str(root))
    first = sum(seen)
    assert first > 0
    seen.clear()

    (root / "memo_bg.md").write_text("# Нова бележка\n\nуникален текст за кеша.\n", encoding="utf-8")
    rag.ingest_project("demo", str(root))
    assert 0 < sum(seen) < first          # only the changed file's chunks were embedded
    res = rag.search_project("demo", "уникален текст за кеша")
    assert any("уникален" in r.excerpt for r in res.results)
