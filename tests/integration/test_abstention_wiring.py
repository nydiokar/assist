"""Abstention suppression wired through search_project (v1 §6.2, F16).

End-to-end on the offline hash+memory path (no model download). Uses the `fusion_score` signal
so the gate runs without a reranker; the signal CHOICE is a test convenience — the measured
production signal is `rerank_score` (EVAL_LOG 2026-10-28). What is tested here is the wiring
contract WP4 depends on, not the corpus threshold:
  - the verdict is always attached to SearchResponse.evidence;
  - with suppress_on_insufficient=true an insufficient verdict withholds passages + warns;
  - with suppression=false the passages are kept and the verdict is advisory.
"""

from __future__ import annotations


def test_verdict_attached_and_passages_suppressed(make_rag, corpus_root):
    # Threshold above any achievable fusion_score (RRF max ≈ 2/(60+1) ≈ 0.033) → always insufficient.
    rag = make_rag(**{
        "retrieval.abstention.enabled": True,
        "retrieval.abstention.signal": "fusion_score",
        "retrieval.abstention.min_top_score": 0.99,
        "retrieval.abstention.suppress_on_insufficient": True,
    })
    rag.ingest_project("demo", corpus_root)
    res = rag.search_project("demo", "What indicators did the approved project define?")
    assert res.evidence.enabled is True
    assert res.evidence.sufficient is False
    assert res.results == []                      # passages withheld
    assert any("insufficient evidence" in w for w in res.warnings)


def test_sufficient_passes_through(make_rag, corpus_root):
    # Threshold of 0 → any positive top score is sufficient; passages flow through.
    rag = make_rag(**{
        "retrieval.abstention.enabled": True,
        "retrieval.abstention.signal": "fusion_score",
        "retrieval.abstention.min_top_score": 0.0,
        "retrieval.abstention.suppress_on_insufficient": True,
    })
    rag.ingest_project("demo", corpus_root)
    res = rag.search_project("demo", "What indicators did the approved project define?")
    assert res.evidence.enabled is True and res.evidence.sufficient is True
    assert res.results                             # not suppressed


def test_suppression_off_keeps_passages_but_flags(make_rag, corpus_root):
    rag = make_rag(**{
        "retrieval.abstention.enabled": True,
        "retrieval.abstention.signal": "fusion_score",
        "retrieval.abstention.min_top_score": 0.99,
        "retrieval.abstention.suppress_on_insufficient": False,
    })
    rag.ingest_project("demo", corpus_root)
    res = rag.search_project("demo", "What indicators did the approved project define?")
    assert res.evidence.enabled is True and res.evidence.sufficient is False
    assert res.results                             # kept: advisory-only mode


def test_gate_off_by_default(make_rag, corpus_root):
    rag = make_rag()
    rag.ingest_project("demo", corpus_root)
    res = rag.search_project("demo", "What indicators did the approved project define?")
    assert res.evidence.enabled is False and res.evidence.sufficient is True
    assert res.results
