"""LexicalIndex port contract."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from project_rag.adapters.sqlite.db import SqliteDb
from project_rag.adapters.sqlite.fts5 import Fts5LexicalIndex
from project_rag.ports.lexical import LexicalIndex
from tests.contracts.conftest import make_chunk


@pytest.fixture(params=["sqlite_fts5"])
def index(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[LexicalIndex]:
    db = SqliteDb(tmp_path / "lex.sqlite3")
    yield Fts5LexicalIndex(db)
    db.close()


def test_search_is_scoped_to_project_and_generation(index: LexicalIndex) -> None:
    index.add_chunks("g1", [make_chunk("c1", "grant number GV-2025-017")])
    index.add_chunks("g2", [make_chunk("c2", "grant number GV-2025-017")])
    index.add_chunks("g1", [make_chunk("c3", "grant number GV-2025-017", project_id="p2")])
    hits = index.search("p1", "g1", "grant", 10)
    assert [h.chunk_id for h in hits] == ["c1"]
    assert hits[0].payload.source_path == "a.md"
    assert hits[0].payload.citation_text == "grant number GV-2025-017"


def test_best_first_and_limit(index: LexicalIndex) -> None:
    index.add_chunks(
        "g1",
        [
            make_chunk("c1", "budget budget budget total"),
            make_chunk("c2", "budget once among many many other words here"),
            make_chunk("c3", "nothing relevant"),
        ],
    )
    hits = index.search("p1", "g1", "budget", 10)
    assert [h.chunk_id for h in hits] == ["c1", "c2"]
    assert [h.score for h in hits] == sorted(h.score for h in hits)
    assert len(index.search("p1", "g1", "budget", 1)) == 1


def test_doc_class_filter(index: LexicalIndex) -> None:
    index.add_chunks(
        "g1",
        [
            make_chunk("c1", "law text youth", doc_class="reference"),
            make_chunk("c2", "project youth"),
        ],
    )
    assert [h.chunk_id for h in index.search("p1", "g1", "youth", 10, "reference")] == ["c1"]


def test_query_operators_are_neutralised(index: LexicalIndex) -> None:
    index.add_chunks("g1", [make_chunk("c1", "alpha beta")])
    for q in ['alpha AND "', "NOT alpha", "alpha* OR (", '""', "   "]:
        index.search("p1", "g1", q, 5)  # must not raise


def test_delete_generation(index: LexicalIndex) -> None:
    index.add_chunks("g1", [make_chunk("c1", "alpha")])
    index.delete_generation("p1", "g1")
    assert index.search("p1", "g1", "alpha", 5) == []
