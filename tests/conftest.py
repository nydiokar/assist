"""Shared fixtures. Builds a ProjectRAG against an isolated temp DB + store per test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project_rag.api import ProjectRAG  # noqa: E402
from project_rag.config import Config  # noqa: E402

FIXTURE_CORPUS = (Path(__file__).resolve().parent / "fixtures" / "synthetic"
                  / "ngo_project_demo")


def _config(tmp_path: Path, **overrides) -> Config:
    cfg = Config()
    cfg.lexical_store.path = str(tmp_path / "manifest.sqlite3")
    cfg.vector_store.persist_path = str(tmp_path / "store")
    for k, v in overrides.items():
        *parents, field = k.split(".")   # supports nested e.g. retrieval.abstention.enabled
        obj = cfg
        for p in parents:
            obj = getattr(obj, p)
        setattr(obj, field, v)
    return cfg


@pytest.fixture
def make_rag(tmp_path):
    created: list[ProjectRAG] = []

    def _factory(**overrides) -> ProjectRAG:
        rag = ProjectRAG(config=_config(tmp_path, **overrides))
        created.append(rag)
        return rag

    yield _factory
    for r in created:
        r.close()


@pytest.fixture
def corpus_root() -> str:
    assert FIXTURE_CORPUS.is_dir(), "run scripts/make_fixtures.py first"
    return str(FIXTURE_CORPUS)
