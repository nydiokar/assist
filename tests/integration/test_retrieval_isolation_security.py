"""Retrieval quality, project isolation, grounding trace, and security (v1 §6.3)."""

from __future__ import annotations

from project_rag.config import RetrievalConfig
from project_rag.retrieval import fuse_and_rank
from project_rag.schema import SourceLocation


def _ingest(rag, root):
    rag.ingest_project("demo", str(root))


def test_exact_identifier_retrieval(make_rag, corpus_root):
    rag = make_rag()
    _ingest(rag, corpus_root)
    res = rag.search_project("demo", "GV-2025-017")
    assert res.results and any("GV-2025-017" in r.excerpt for r in res.results)


def test_paraphrase_retrieval(make_rag, corpus_root):
    rag = make_rag()
    _ingest(rag, corpus_root)
    res = rag.search_project("demo", "what metrics did the project commit to")
    assert any(r.source_path in {"proposal.md", "indicators.xlsx"} for r in res.results)


def test_bulgarian_retrieval(make_rag, corpus_root):
    rag = make_rag()
    _ingest(rag, corpus_root)
    res = rag.search_project("demo", "обучение по земеделие през март")
    assert any(r.source_path == "memo_bg.md" for r in res.results)


def test_table_evidence_retrieval(make_rag, corpus_root):
    rag = make_rag()
    _ingest(rag, corpus_root)
    res = rag.search_project("demo", "reported_actual target indicator people")
    assert any(r.source_path == "indicators.xlsx" for r in res.results)


def test_empty_result_behavior(make_rag, corpus_root):
    rag = make_rag()
    _ingest(rag, corpus_root)
    res = rag.search_project("demo", "quantum chromodynamics lattice gauge theory")
    # may return weak candidates, but must never raise and must carry a valid trace
    assert res.index_generation
    assert rag.traces[-1].result_count == len(res.results)


def test_no_generation_returns_empty_with_warning(make_rag):
    rag = make_rag()
    res = rag.search_project("ghost", "anything")
    assert res.results == []
    assert any("no active generation" in w for w in res.warnings)


def test_project_isolation_zero_leakage(make_rag, corpus_root, tmp_path):
    rag = make_rag()
    rag.ingest_project("proj_a", corpus_root)
    # second project with distinct unique content
    other = tmp_path / "other"
    other.mkdir()
    (other / "secret.md").write_text(
        "# Secret\n\nZEBRAPHONE classified marker only in B.\n", encoding="utf-8"
    )
    rag.ingest_project("proj_b", str(other))
    res_a = rag.search_project("proj_a", "ZEBRAPHONE classified marker")
    assert all("ZEBRAPHONE" not in r.excerpt for r in res_a.results)
    res_b = rag.search_project("proj_b", "ZEBRAPHONE classified marker")
    assert any("ZEBRAPHONE" in r.excerpt for r in res_b.results)


def test_grounding_trace_recorded(make_rag, corpus_root):
    rag = make_rag()
    _ingest(rag, corpus_root)
    res = rag.search_project("demo", "reporting deadline")
    trace = rag.traces[-1]
    assert trace.tool == "search_project"
    assert trace.query_hash and "reporting" not in trace.model_dump_json()  # query not logged raw
    assert trace.retrieved_chunk_ids == [r.chunk_id for r in res.results]


def test_open_source_path_traversal_rejected(make_rag, corpus_root):
    rag = make_rag()
    _ingest(rag, corpus_root)
    rag.register_project_root("demo", corpus_root)
    resp = rag.open_source("demo", "doc", source_path="../../../../Windows/System32/x.txt")
    assert resp.excerpt == ""
    assert any("escape" in w or "not found" in w for w in resp.warnings)


def test_open_source_returns_original_text(make_rag, corpus_root):
    rag = make_rag()
    _ingest(rag, corpus_root)
    rag.register_project_root("demo", corpus_root)
    resp = rag.open_source("demo", "doc", source_path="proposal.md")
    assert "GV-2025-017" in resp.excerpt


def test_query_length_bounded(make_rag, corpus_root):
    rag = make_rag(**{"security.max_query_chars": 10})
    _ingest(rag, corpus_root)
    res = rag.search_project("demo", "x" * 500)
    assert any("truncated" in w for w in res.warnings)


def test_fusion_is_deterministic():
    cfg = RetrievalConfig()
    payload = {
        "source_path": "a.md",
        "document_id": "d",
        "citation_text": "t",
        "location": SourceLocation(),
    }
    dense = [("ck_1", 0.1, payload), ("ck_2", 0.2, payload)]
    lexical = [("ck_2", 1.0, payload), ("ck_3", 2.0, payload)]
    a = [r.chunk_id for r in fuse_and_rank(dense, lexical, cfg)]
    b = [r.chunk_id for r in fuse_and_rank(dense, lexical, cfg)]
    assert a == b
    # ck_2 appears in both channels → should rank first under RRF
    assert a[0] == "ck_2"
