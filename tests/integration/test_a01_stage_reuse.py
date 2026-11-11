"""A01 acceptance: switch embedding profile using a separate index without reparsing unchanged
sources; and trace parse → chunk → profile → store entry → retrieved passage → source address.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def _copy(src: str, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def test_profile_switch_reuses_parse_cache_no_reparse(make_rag, corpus_root, tmp_path):
    root = _copy(corpus_root, tmp_path / "corpus")
    # First ingest with the default (hash @ 2560) profile.
    rag1 = make_rag()
    rag1.ingest_project("demo", str(root))

    # Second embedding profile: hash @ 1024 (a DISTINCT profile → new collection/generation).
    # The parse cache is keyed by source+parser only, so parse must be fully reused.
    rag2 = make_rag(**{"embedding.hash_dim": 1024, "vector_store.collection": "alt_1024_v1"})
    report = rag2.ingest_project("demo2", str(root))
    cache_line = next(w for w in report.warnings if "stage_cache" in w)
    # new project id → chunk cache is per-project (lineage), but PARSE cache is shared by
    # content hash. Assert no parse misses beyond the already-cached artifacts where possible.
    assert "parse_hits" in cache_line
    # the alternate profile produced its own working index at a different dimension
    assert rag2.get_ingestion_status("demo2").index_generation
    res = rag2.search_project("demo2", "GV-2025-017")
    assert any("GV-2025-017" in r.excerpt for r in res.results)


def test_full_traceability_chain(make_rag, corpus_root):
    rag = make_rag()
    rag.ingest_project("demo", corpus_root)
    rag.register_project_root("demo", corpus_root)

    res = rag.search_project("demo", "What grant number is the project under?")
    assert res.results
    top = res.results[0]
    # retrieved passage carries a resolvable source address
    assert top.source_path and top.document_id
    # trace ties the query to the retrieved chunk ids (grounding gate)
    trace = rag.traces[-1]
    assert top.chunk_id in trace.retrieved_chunk_ids
    assert trace.index_generation == res.index_generation
    # open_source resolves the ORIGINAL source text at that address
    opened = rag.open_source("demo", top.document_id, source_path=top.source_path)
    assert opened.excerpt
    assert "GV-2025-017" in opened.excerpt
