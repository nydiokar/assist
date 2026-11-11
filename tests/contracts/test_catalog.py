"""Catalog port contract."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from project_rag.adapters.sqlite.catalog import SqliteCatalog
from project_rag.adapters.sqlite.db import SqliteDb
from project_rag.domain.config import EmbeddingConfig
from project_rag.domain.models import DocumentRecord
from project_rag.ports.catalog import Catalog


@pytest.fixture(params=["sqlite"])
def catalog(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Catalog]:
    db = SqliteDb(tmp_path / "catalog.sqlite3")
    yield SqliteCatalog(db)
    db.close()


def _doc(path: str, gen: str, n: int = 1) -> DocumentRecord:
    return DocumentRecord(
        project_id="p1",
        source_id="src_" + path,
        source_path=path,
        content_hash="h_" + path,
        version_id="v1",
        state="READY",
        generation=gen,
        n_chunks=n,
        diagnostics="",
    )


def test_profile_pin_and_mismatch(catalog: Catalog) -> None:
    hash_profile = EmbeddingConfig(provider="hash").profile()
    assert catalog.check_compatibility("p1", hash_profile, "col") == []
    catalog.set_project_profile("p1", hash_profile, "col")
    assert catalog.check_compatibility("p1", hash_profile, "col") == []
    other = EmbeddingConfig(provider="hash", hash_dim=128).profile()
    assert any("profile mismatch" in p for p in catalog.check_compatibility("p1", other, "col"))
    assert any(
        "collection mismatch" in p for p in catalog.check_compatibility("p1", hash_profile, "x")
    )


def test_publish_and_documents_roundtrip(catalog: Catalog) -> None:
    assert catalog.active_generation("p1") is None
    catalog.upsert_document(_doc("a.md", "g1", 2))
    catalog.upsert_document(_doc("b.md", "g1"))
    catalog.record_chunks("p1", "g1", [("c1", "src_a.md", "v1"), ("c2", "src_a.md", "v1")])
    catalog.publish_generation("p1", "g1")
    assert catalog.active_generation("p1") == "g1"
    assert sorted(d.source_path for d in catalog.document_states("p1", "g1")) == ["a.md", "b.md"]
    assert catalog.document_states("p1", "g1")[0] in {_doc("a.md", "g1", 2), _doc("b.md", "g1")}
    assert catalog.prior_hashes("p1", "g1")["a.md"] == ("h_a.md", "src_a.md")
    assert sorted(catalog.chunk_ids_for_generation("p1", "g1")) == ["c1", "c2"]


def test_delete_generation_removes_its_rows(catalog: Catalog) -> None:
    catalog.upsert_document(_doc("a.md", "g1"))
    catalog.record_chunks("p1", "g1", [("c1", "src_a.md", "v1")])
    catalog.delete_generation("p1", "g1")
    assert catalog.document_states("p1", "g1") == []
    assert catalog.chunk_ids_for_generation("p1", "g1") == []
