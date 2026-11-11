"""CorpusReader port contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from project_rag.adapters.corpus_local_fs import LocalFsCorpus
from project_rag.domain.errors import PathOutsideRoot
from project_rag.ports.corpus import CorpusReader


@pytest.fixture(params=[pytest.param(LocalFsCorpus(), id="local_fs")])
def corpus(request: pytest.FixtureRequest) -> CorpusReader:
    return request.param


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "sub").mkdir(parents=True)
    (root / "a.md").write_text("alpha", encoding="utf-8")
    (root / "sub" / "b.CSV").write_text("h\n1\n", encoding="utf-8")
    (root / "sub" / "photo.jpg").write_bytes(b"\xff\xd8")
    (root / "~$lock.docx").write_bytes(b"x")  # Office lock file: ignored
    (root / "draft.tmp").write_bytes(b"x")  # temp file: ignored
    return root


def test_lists_sorted_relative_hashed_and_classified(corpus: CorpusReader, tree: Path) -> None:
    files = corpus.list_files(str(tree))
    assert [f.source_path for f in files] == ["a.md", "sub/b.CSV", "sub/photo.jpg"]
    by = {f.source_path: f for f in files}
    assert by["a.md"].content_hash == hashlib.sha256(b"alpha").hexdigest()
    assert by["sub/b.CSV"].ext == ".csv" and by["sub/b.CSV"].supported
    assert not by["sub/photo.jpg"].supported
    assert all(corpus.is_file(f.abs_path) for f in files)


def test_resolve_stays_inside_root(corpus: CorpusReader, tree: Path) -> None:
    inside = corpus.resolve(str(tree), "sub/b.CSV")
    assert corpus.is_file(inside)
    with pytest.raises(PathOutsideRoot):
        corpus.resolve(str(tree), "../outside.txt")


def test_missing_root_is_an_error(corpus: CorpusReader, tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        corpus.list_files(str(tmp_path / "absent"))
