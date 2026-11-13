"""Exact-duplicate collapse (S5, DECISIONS 2026-11-13) and per-occurrence source paths (F24)."""

from __future__ import annotations

import shutil
from pathlib import Path

from project_rag.domain.models import DocState

_TEXT = "# Grant\n\nThe zebracorn initiative grant number is ZX-4242 for the youth council.\n"
_QUERY = "zebracorn initiative grant number"


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "Resources").mkdir(parents=True)
    (root / "Meeting").mkdir()
    (root / "other.md").write_text("# Other\n\nUnrelated text about budgets.\n", encoding="utf-8")
    (root / "Meeting" / "grant.md").write_text(_TEXT, encoding="utf-8")
    (root / "Meeting" / "Copy of grant.md").write_text(_TEXT, encoding="utf-8")
    (root / "Resources" / "grant.md").write_text(_TEXT, encoding="utf-8")
    return root


def _states(report) -> dict[str, DocState]:
    return {f.source_path: f.state for f in report.files}


def test_exact_duplicates_are_recorded_and_only_the_canonical_is_indexed(make_rag, tmp_path):
    root = _corpus(tmp_path)
    rag = make_rag(**{"classification.reference_prefixes": ["Resources/"]})
    r = rag.ingest_project("demo", str(root))
    states = _states(r)
    # Canonical: not a "Copy of" name, not reference material.
    assert states["Meeting/grant.md"] == DocState.READY
    assert states["Meeting/Copy of grant.md"] == DocState.DUPLICATE
    assert states["Resources/grant.md"] == DocState.DUPLICATE
    assert r.run is not None
    assert r.run.duplicate_of == {
        "Meeting/Copy of grant.md": "Meeting/grant.md",
        "Resources/grant.md": "Meeting/grant.md",
    }
    assert r.run.counts_by_state["DUPLICATE"] == 2
    assert r.run.coverage_pct == 100.0  # 2 distinct documents, both READY
    hits = rag.search_project("demo", _QUERY).results
    assert {h.source_path for h in hits if "ZX-4242" in h.excerpt} == {"Meeting/grant.md"}
    # Idempotent with duplicates present.
    r2 = rag.ingest_project("demo", str(root))
    assert r2.index_generation == r.index_generation
    assert _states(r2)["Resources/grant.md"] == DocState.DUPLICATE


def test_removing_the_canonical_promotes_a_remaining_copy(make_rag, tmp_path):
    root = _corpus(tmp_path)
    rag = make_rag(**{"classification.reference_prefixes": ["Resources/"]})
    rag.ingest_project("demo", str(root))
    (root / "Meeting" / "grant.md").unlink()
    r2 = rag.ingest_project("demo", str(root))
    states = _states(r2)
    assert states["Meeting/Copy of grant.md"] == DocState.DUPLICATE
    assert states["Resources/grant.md"] == DocState.READY
    hits = rag.search_project("demo", _QUERY).results
    assert {h.source_path for h in hits if "ZX-4242" in h.excerpt} == {"Resources/grant.md"}


def test_without_collapse_each_copy_cites_its_own_path(make_rag, tmp_path):
    """F24: the parse cache is content-keyed; a same-bytes copy must not inherit the first
    occurrence's source_path."""
    root = _corpus(tmp_path)
    rag = make_rag(**{"ingest.collapse_exact_duplicates": False})
    r = rag.ingest_project("demo", str(root))
    assert all(f.state == DocState.READY for f in r.files)
    hits = rag.search_project("demo", _QUERY).results
    cited = {h.source_path for h in hits if "ZX-4242" in h.excerpt}
    assert cited == {"Meeting/grant.md", "Meeting/Copy of grant.md", "Resources/grant.md"}
    for h in hits:
        src = rag.open_source("demo", h.document_id, source_path=h.source_path)
        assert src.source_path == h.source_path


def test_renamed_file_cites_the_new_path(make_rag, tmp_path):
    """F24 reproducer: rename between runs → parse-cache hit must cite the new name."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "old name.md").write_text(_TEXT, encoding="utf-8")
    rag = make_rag()
    rag.ingest_project("demo", str(root))
    shutil.move(root / "old name.md", root / "new name.md")
    rag.ingest_project("demo", str(root))
    hits = rag.search_project("demo", _QUERY).results
    assert hits and {h.source_path for h in hits} == {"new name.md"}


def test_curation_list_excludes_exact_paths(make_rag, tmp_path):
    root = _corpus(tmp_path)
    rag = make_rag(**{"ingest.exclude_paths": ["other.md"]})
    r = rag.ingest_project("demo", str(root))
    assert "other.md" not in _states(r)
    assert r.run is not None and r.run.n_excluded == 1
