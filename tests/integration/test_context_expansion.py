"""Parent-section context expansion wired through search_project (v1 §7, Task A).

Offline hash+memory path (no model download). Verifies the query-time contract a WP4 answerer
relies on, not corpus metrics (those are measured in EVAL_LOG via scripts/probe_span_coverage.py):
  - off by default → expanded_context stays empty, behavior unchanged;
  - on → each served result carries parent-section text that is a SUPERSET of its own excerpt;
  - expansion recovers a sibling fact that is in the chunk's section but not its top excerpt;
  - suppressed (insufficient-evidence) responses are not expanded (nothing to widen).
"""

from __future__ import annotations


def test_expansion_off_by_default(make_rag, corpus_root):
    rag = make_rag()
    rag.ingest_project("demo", corpus_root)
    res = rag.search_project("demo", "What indicators did the approved project define?")
    assert res.results
    assert all(r.expanded_context == "" for r in res.results)


def test_expansion_superset_of_excerpt(make_rag, corpus_root):
    rag = make_rag(**{"retrieval.context_expansion.enabled": True})
    rag.ingest_project("demo", corpus_root)
    res = rag.search_project("demo", "What indicators did the approved project define?")
    assert res.results
    for r in res.results:
        # expansion never loses the verbatim chunk text; it only widens it
        assert r.expanded_context
        assert r.excerpt.strip() in r.expanded_context


def test_expansion_recovers_section_sibling_fact(make_rag, corpus_root):
    # "Project summary" section holds both the date range and grant number GV-2025-017. A chunk
    # retrieved for the summary should expose the grant number via section expansion even if the
    # top excerpt was truncated/focused elsewhere.
    rag = make_rag(**{"retrieval.context_expansion.enabled": True})
    rag.ingest_project("demo", corpus_root)
    res = rag.search_project("demo", "When does the Green Villages project run?")
    assert res.results
    joined = " ".join(r.expanded_context for r in res.results)
    assert "GV-2025-017" in joined


def test_suppressed_response_not_expanded(make_rag, corpus_root):
    rag = make_rag(**{
        "retrieval.context_expansion.enabled": True,
        "retrieval.abstention.enabled": True,
        "retrieval.abstention.signal": "fusion_score",
        "retrieval.abstention.min_top_score": 0.99,     # always insufficient
        "retrieval.abstention.suppress_on_insufficient": True,
    })
    rag.ingest_project("demo", corpus_root)
    res = rag.search_project("demo", "What indicators did the approved project define?")
    assert res.results == []                            # withheld, nothing to expand
