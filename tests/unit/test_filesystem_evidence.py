"""Filesystem-presence chunk builder (Task 3; DECISIONS 2026-10-28).

Pure function over a DiscoveredFile list. Verifies: one chunk per folder that has an unparseable
file; parseable-only folders are skipped; the not-content banner and doc_class=filesystem tag are
present; filenames are listed (capped); ids are deterministic and change when the inventory does.
"""

from __future__ import annotations

from project_rag.core.filesystem_evidence import FILESYSTEM_DOC_CLASS, build_filesystem_chunks
from project_rag.domain.models import DiscoveredFile


def _df(path: str, supported: bool, chash: str = "h") -> DiscoveredFile:
    ext = ("." + path.rsplit(".", 1)[1]) if "." in path.rsplit("/", 1)[-1] else ""
    return DiscoveredFile(
        source_path=path,
        abs_path="/abs/" + path,
        ext=ext,
        size=1,
        mtime=0.0,
        content_hash=chash,
        supported=supported,
    )


def test_one_chunk_per_folder_with_unparseable_only() -> None:
    files = [
        _df("Уебинари/2. Младежко участие на практика/clip.mp4", False),
        _df("Уебинари/2. Младежко участие на практика/thumb.jpg", False),
        _df("Уебинари/2. Младежко участие на практика/notes.docx", True),
        _df("Docs/report.docx", True),  # parseable-only folder → skipped
        _df("Images/a.png", False),  # unparseable-only folder → included
    ]
    chunks = build_filesystem_chunks("ypq", files)
    folders = {c.heading for c in chunks}
    assert folders == {"Уебинари/2. Младежко участие на практика", "Images"}
    assert all(c.doc_class == FILESYSTEM_DOC_CLASS for c in chunks)
    assert all(c.source_type == "filesystem" for c in chunks)


def test_chunk_text_is_labelled_and_lists_names_and_counts() -> None:
    files = [
        _df("Уебинари/2. Младежко участие на практика/clip.mp4", False),
        _df("Уебинари/2. Младежко участие на практика/thumb.jpg", False),
        _df("Уебинари/2. Младежко участие на практика/notes.docx", True),
    ]
    c = build_filesystem_chunks("ypq", files)[0]
    assert "NOT DOCUMENT CONTENT" in c.embed_text
    assert "NOT indexed" in c.embed_text
    assert "Уебинари/2. Младежко участие на практика" in c.embed_text
    assert "clip.mp4" in c.embed_text and "thumb.jpg" in c.embed_text
    assert "2 unparseable" in c.embed_text and "1 parseable" in c.embed_text
    # citation == embed for presence chunks (nothing is enriched away)
    assert c.citation_text == c.embed_text


def test_names_are_capped() -> None:
    files = [_df(f"Big/{i}.png", False) for i in range(50)]
    c = build_filesystem_chunks("ypq", files)[0]
    assert "…" in c.embed_text  # truncation marker present
    assert "50 unparseable" in c.embed_text  # count still reflects all 50


def test_ids_deterministic_and_inventory_sensitive() -> None:
    base = [_df("F/a.png", False), _df("F/b.png", False)]
    c1 = build_filesystem_chunks("ypq", base)[0]
    c2 = build_filesystem_chunks("ypq", list(base))[0]
    assert c1.chunk_id == c2.chunk_id  # deterministic
    c3 = build_filesystem_chunks("ypq", [*base, _df("F/c.png", False)])[0]
    assert c3.version_id != c1.version_id  # adding a file changes the version
    assert c3.chunk_id != c1.chunk_id


def test_no_unparseable_means_no_chunks() -> None:
    assert build_filesystem_chunks("ypq", [_df("D/x.docx", True)]) == []
