"""doc_class filter (project_record vs reference) and per-channel baselines (v1 §6.1)."""

from __future__ import annotations


def test_reference_docs_filterable(make_rag, corpus_root):
    rag = make_rag(**{"classification.reference_prefixes": ["memo_bg"]})
    rag.ingest_project("demo", corpus_root)
    q = "Обучението по устойчиво земеделие март участници"
    allr = rag.search_project("demo", q, top_k=20)
    rec = rag.search_project("demo", q, top_k=20, filters={"doc_class": "project_record"})
    ref = rag.search_project("demo", q, top_k=20, filters={"doc_class": "reference"})
    assert any(r.source_path == "memo_bg.md" for r in allr.results)
    assert rec.results and all(r.source_path != "memo_bg.md" for r in rec.results)
    assert all(r.source_path == "memo_bg.md" for r in ref.results)


def test_lexical_only_channel_and_unknown_filter_warns(make_rag, corpus_root):
    rag = make_rag()
    rag.ingest_project("demo", corpus_root)
    res = rag.search_project("demo", "GV-2025-017", filters={"channels": ["lexical"], "x": 1})
    assert res.results and all(r.retrieval_channels == ["lexical"] for r in res.results)
    assert any("unsupported filter ignored: x" in w for w in res.warnings)
