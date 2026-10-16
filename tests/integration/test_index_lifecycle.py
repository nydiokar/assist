"""Index lifecycle: idempotency, modify, delete, shrink, mismatch refusal (v1 §6.3)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


def _copy_corpus(src: str, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def test_ingest_twice_no_duplicate_chunks(make_rag, corpus_root, tmp_path):
    root = _copy_corpus(corpus_root, tmp_path / "corpus")
    rag = make_rag()
    r1 = rag.ingest_project("demo", str(root))
    gen1 = r1.index_generation
    r2 = rag.ingest_project("demo", str(root))
    assert r2.index_generation == gen1
    assert r2.unchanged == len(r2.files)
    assert r2.added == 0 and r2.modified == 0
    status = rag.get_ingestion_status("demo")
    assert status.n_chunks == rag.get_ingestion_status("demo").n_chunks  # stable


def test_modified_file_removes_old_chunks(make_rag, corpus_root, tmp_path):
    root = _copy_corpus(corpus_root, tmp_path / "corpus")
    rag = make_rag()
    rag.ingest_project("demo", str(root))
    before = rag.search_project("demo", "unicorn magic sparkle keyword")
    assert all("unicorn" not in r.excerpt.lower() for r in before.results)
    (root / "proposal.md").write_text("# New\n\nunicorn magic sparkle keyword content.\n",
                                      encoding="utf-8")
    r2 = rag.ingest_project("demo", str(root))
    assert r2.modified == 1
    after = rag.search_project("demo", "unicorn magic sparkle keyword")
    assert any("unicorn" in r.excerpt.lower() for r in after.results)
    # old proposal indicator text gone from active set
    gone = rag.search_project("demo", "approved project defines the following indicators")
    assert all("defines the following indicators" not in r.excerpt for r in gone.results)


def test_deleted_source_absent_from_active_set(make_rag, corpus_root, tmp_path):
    root = _copy_corpus(corpus_root, tmp_path / "corpus")
    rag = make_rag()
    rag.ingest_project("demo", str(root))
    (root / "memo_bg.md").unlink()
    r2 = rag.ingest_project("demo", str(root))
    assert r2.deleted == 1
    res = rag.search_project("demo", "Обучението по устойчиво земеделие")
    assert all(x.source_path != "memo_bg.md" for x in res.results)


def test_shrunk_document_leaves_no_orphans(make_rag, tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("# H\n\nalpha beta gamma delta epsilon zeta eta theta.\n",
                               encoding="utf-8")
    rag = make_rag()
    rag.ingest_project("p", str(root))
    (root / "a.md").write_text("# H\n\nalpha.\n", encoding="utf-8")
    rag.ingest_project("p", str(root))
    res = rag.search_project("p", "zeta eta theta")
    assert all("zeta" not in r.excerpt for r in res.results)


def test_embedding_profile_mismatch_refused(make_rag, corpus_root, tmp_path):
    root = _copy_corpus(corpus_root, tmp_path / "corpus")
    rag = make_rag()
    rag.ingest_project("demo", str(root))
    rag2 = make_rag(**{"embedding.provider": "hash", "embedding.hash_dim": 128})
    report = rag2.ingest_project("demo", str(root))
    assert any("REFUSED" in w for w in report.warnings)


def test_dry_run_writes_nothing(make_rag, corpus_root, tmp_path):
    root = _copy_corpus(corpus_root, tmp_path / "corpus")
    rag = make_rag()
    report = rag.ingest_project("demo", str(root), dry_run=True)
    assert report.dry_run
    assert rag.get_ingestion_status("demo").index_generation is None


def test_chunking_config_change_forces_new_generation(make_rag, corpus_root, tmp_path):
    rag = make_rag()
    g1 = rag.ingest_project("demo", corpus_root).index_generation
    rag._cfg.chunking.target_tokens = 120
    rag._indexer._cfg.chunking.target_tokens = 120
    r2 = rag.ingest_project("demo", corpus_root)
    assert r2.index_generation != g1 and "no changes" not in " ".join(r2.warnings)
