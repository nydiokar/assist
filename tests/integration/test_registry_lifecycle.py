"""S4 exit checks on fixtures: rename keeps project_id with no re-index; a deleted file leaves
search; a second unchanged ingest computes no embeddings and keeps the generation id."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from project_rag.core.slug import fallback_drive_key, slugify, unique_slug
from project_rag.domain.config import Config, with_overrides
from project_rag.services.registry import ingest_corpus
from project_rag.wiring import Services, build
from tests.conftest import FIXTURE_CORPUS


def _services(tmp: Path) -> Services:
    cfg = with_overrides(
        Config(),
        {
            "lexical_store.path": str(tmp / "state" / "store.sqlite3"),
            "vector_store.persist_path": str(tmp / "state" / "vectors"),
        },
    )
    return build(cfg)


def _write_manifest(root: Path, folders: dict[str, str]) -> None:
    (root / ".sync").mkdir(exist_ok=True)
    payload = {"version": 1, "folders": [{"path": p, "drive_id": d} for p, d in folders.items()]}
    (root / ".sync" / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _corpus_root(tmp: Path) -> Path:
    root = tmp / "drive"
    shutil.copytree(FIXTURE_CORPUS, root / "Зелени села")  # Cyrillic folder name
    shutil.copytree(FIXTURE_CORPUS, root / "Second Project")
    _write_manifest(root, {"Зелени села": "drv-green-1", "Second Project": "drv-second-2"})
    return root


def test_rename_keeps_project_id_and_does_not_reindex(tmp_path: Path) -> None:
    root = _corpus_root(tmp_path)
    s = _services(tmp_path)
    first = {
        rec.identity.drive_id: (rec, rep)
        for rec, rep in ingest_corpus(s.registry, s.ingest, str(root))
    }
    green, green_report = first["drv-green-1"]
    assert green.identity.project_id == "zeleni-sela" and green.key_source == "drive"
    assert first["drv-second-2"][0].identity.project_id == "second-project"
    gen = green_report.index_generation

    (root / "Зелени села").rename(root / "Green Villages 2025")
    _write_manifest(root, {"Green Villages 2025": "drv-green-1", "Second Project": "drv-second-2"})
    second = {
        rec.identity.drive_id: (rec, rep)
        for rec, rep in ingest_corpus(s.registry, s.ingest, str(root))
    }
    rec, rep = second["drv-green-1"]
    assert rec.identity.project_id == "zeleni-sela"  # same id
    assert rec.identity.display_name == "Green Villages 2025"
    assert rec.renamed_from == "Зелени села"
    assert rep.index_generation == gen  # same generation
    assert "no changes: active generation unchanged" in rep.warnings  # no re-index
    assert rep.run is not None and rep.run.embeddings_computed == 0
    names = {p.project_id: p.display_name for p in s.catalog.list_projects()}
    assert names == {"zeleni-sela": "Green Villages 2025", "second-project": "Second Project"}
    s.close()


def test_without_sidecar_the_fallback_key_is_the_folder_hash(tmp_path: Path) -> None:
    root = tmp_path / "drive"
    shutil.copytree(FIXTURE_CORPUS, root / "Demo")
    s = _services(tmp_path)
    (rec,) = s.registry.sync(str(root))
    assert rec.key_source == "path-fallback"
    assert rec.identity.drive_id == fallback_drive_key("Demo")
    s.close()


def test_deleted_file_is_absent_after_next_ingest(tmp_path: Path) -> None:
    root = _corpus_root(tmp_path)
    s = _services(tmp_path)
    ingest_corpus(s.registry, s.ingest, str(root))
    query = "When is the final report due and what attachments are required?"
    before = s.search.search_project("zeleni-sela", query, top_k=8)
    assert any(r.source_path == "proposal.md" for r in before.results)
    (root / "Зелени села" / "proposal.md").unlink()
    results = {
        rec.identity.project_id: rep for rec, rep in ingest_corpus(s.registry, s.ingest, str(root))
    }
    assert results["zeleni-sela"].deleted == 1
    after = s.search.search_project("zeleni-sela", query, top_k=8)
    assert all(r.source_path != "proposal.md" for r in after.results)
    gen = s.catalog.active_generation("zeleni-sela")
    assert gen is not None
    assert "proposal.md" not in {
        d.source_path for d in s.catalog.document_states("zeleni-sela", gen)
    }
    # The other project is untouched.
    other = s.search.search_project("second-project", query, top_k=8)
    assert any(r.source_path == "proposal.md" for r in other.results)
    s.close()


def test_second_unchanged_ingest_creates_no_embeddings(tmp_path: Path) -> None:
    root = _corpus_root(tmp_path)
    s = _services(tmp_path)
    first = {
        r.identity.project_id: rep for r, rep in ingest_corpus(s.registry, s.ingest, str(root))
    }
    again = {
        r.identity.project_id: rep for r, rep in ingest_corpus(s.registry, s.ingest, str(root))
    }
    for pid in ("zeleni-sela", "second-project"):
        assert again[pid].index_generation == first[pid].index_generation
        assert again[pid].run is not None and again[pid].run.embeddings_computed == 0
    # Folders sync in sorted order ("Second Project" < "Зелени села"): the first computes the
    # vectors, the second shares every chunk text and finds them all in the embedding cache.
    computed = {pid: rep.run.embeddings_computed for pid, rep in first.items() if rep.run}
    assert computed["second-project"] > 0 and computed["zeleni-sela"] == 0
    s.close()


def test_slugs() -> None:
    assert slugify("Младежко участие") == "mladezhko-uchastie"
    assert slugify("  Youth Participation (2025)! ") == "youth-participation-2025"
    assert slugify("???") == "project"
    assert unique_slug("Demo", {"demo", "demo-2"}) == "demo-3"


def test_rclone_lsjson_sidecar_gives_drive_ids_and_links(tmp_path: Path) -> None:
    root = tmp_path / "drive"
    shutil.copytree(FIXTURE_CORPUS, root / "Green")
    (root / ".sync").mkdir()
    listing = [
        {"Path": "Green", "Name": "Green", "IsDir": True, "ID": "1FolderId"},
        {"Path": "Green/proposal.md", "Name": "proposal.md", "IsDir": False, "ID": "1FileId"},
        {"Path": "Green/memo_bg.md", "Name": "memo_bg.md", "IsDir": False},  # no ID: skipped
    ]
    (root / ".sync" / "lsjson.json").write_text(json.dumps(listing), encoding="utf-8")
    s = _services(tmp_path)
    (rec,) = s.registry.sync(str(root))
    assert rec.key_source == "drive" and rec.identity.drive_id == "1FolderId"
    manifest = s.registry.read_manifest(str(root))
    assert manifest is not None
    assert [(f.path, f.web_link) for f in manifest.files] == [
        ("Green/proposal.md", "https://drive.google.com/open?id=1FileId")
    ]
    s.close()
