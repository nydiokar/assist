"""Reranker port contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from project_rag.adapters.rerank_identity import IdentityReranker
from project_rag.domain.config import RerankConfig
from project_rag.ports.reranker import Reranker


def _identity() -> Reranker:
    return IdentityReranker()


def _torch() -> Reranker:
    pytest.importorskip("sentence_transformers")
    from project_rag.adapters.rerank_torch import CrossEncoderReranker

    return CrossEncoderReranker(RerankConfig(enabled=True))


def _openai_compat() -> Reranker:
    from project_rag.adapters.rerank_openai_compat import OpenAICompatReranker
    from project_rag.wiring import load_config

    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "dev_llama_server.yaml")
    r = OpenAICompatReranker(cfg.retrieval.rerank)
    if not r.healthy():
        pytest.skip("llama-server rerank not reachable (make models-up)")
    return r


ADAPTERS: list[object] = [
    pytest.param(_identity, id="identity"),
    pytest.param(_openai_compat, id="openai_compat", marks=pytest.mark.live),
    pytest.param(_torch, id="torch_bge", marks=pytest.mark.live),
]


@pytest.fixture(params=ADAPTERS)
def reranker(request: pytest.FixtureRequest) -> Reranker:
    factory: Callable[[], Reranker] = request.param
    return factory()


def test_one_float_per_passage(reranker: Reranker) -> None:
    passages = ["бюджет 49 697 EUR", "unrelated text about weather", "the total budget"]
    scores = reranker.score("what is the total budget?", passages)
    assert len(scores) == len(passages)
    assert all(isinstance(s, float) for s in scores)


def test_empty_passages_score_nothing(reranker: Reranker) -> None:
    assert reranker.score("q", []) == []


def test_deterministic(reranker: Reranker) -> None:
    passages = ["a b c", "d e f"]
    assert reranker.score("q", passages) == reranker.score("q", passages)
