"""HTTP provider adapters against a stub server (respx): request shape, batching, the client-side
query prompt, normalisation, token-count cache, retries and port-error translation (S3, A1-A3)."""

from __future__ import annotations

import json
import math

import httpx
import numpy as np
import pytest
import respx

from project_rag.adapters.embed_openai_compat import OpenAICompatEmbedder
from project_rag.adapters.rerank_openai_compat import OpenAICompatReranker
from project_rag.domain.config import (
    EmbeddingConfig,
    OpenAICompatEmbedConfig,
    OpenAICompatRerankConfig,
    RerankConfig,
    RetryPolicy,
)
from project_rag.domain.errors import (
    ProviderContractViolation,
    ProviderTimeout,
    ProviderUnavailable,
)

BASE = "http://embed.test"
RBASE = "http://rerank.test"
DIM = 4
FAST = RetryPolicy(attempts=3, backoff_s=0.0, max_backoff_s=0.0)
PROMPT = OpenAICompatEmbedConfig().query_prompt


def _embedder(**over: object) -> OpenAICompatEmbedder:
    o = OpenAICompatEmbedConfig(base_url=BASE, dim=DIM, retry=FAST, max_batch=2, **over)  # type: ignore[arg-type]
    return OpenAICompatEmbedder(EmbeddingConfig(provider="openai_compat", openai_compat=o))


def _vec(seed: int) -> list[float]:
    return [float(seed + 1), 2.0, 0.0, 1.0]


def _echo_embeddings(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    data = [{"index": i, "embedding": _vec(i)} for i, _ in enumerate(body["input"])]
    return httpx.Response(200, json={"data": list(reversed(data))})  # out of order on purpose


@respx.mock
def test_passages_batched_unprefixed_ordered_and_normalised() -> None:
    route = respx.post(f"{BASE}/v1/embeddings").mock(side_effect=_echo_embeddings)
    mat = _embedder().embed_passages(["a", "b", "c", "d", "e"])
    assert route.call_count == 3  # max_batch=2
    sent = [json.loads(c.request.content)["input"] for c in route.calls]
    assert sent == [["a", "b"], ["c", "d"], ["e"]]
    assert mat.shape == (5, DIM) and mat.dtype == np.float32
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0)
    expected0 = np.asarray(_vec(0)) / np.linalg.norm(_vec(0))
    assert np.allclose(mat[0], expected0)  # index order restored


@respx.mock
def test_query_prompt_is_applied_client_side_to_queries_only() -> None:
    route = respx.post(f"{BASE}/v1/embeddings").mock(side_effect=_echo_embeddings)
    e = _embedder()
    e.embed_queries(["what is the budget?"])
    e.embed_passages(["what is the budget?"])
    q_input = json.loads(route.calls[0].request.content)["input"][0]
    p_input = json.loads(route.calls[1].request.content)["input"][0]
    assert q_input == PROMPT + "what is the budget?"
    assert PROMPT.endswith("\nQuery:") and PROMPT.startswith("Instruct: ")
    assert p_input == "what is the budget?"


@respx.mock
def test_append_text_goes_on_every_input() -> None:
    route = respx.post(f"{BASE}/v1/embeddings").mock(side_effect=_echo_embeddings)
    _embedder(append_text="<|endoftext|>").embed_queries(["q"])
    assert json.loads(route.calls[0].request.content)["input"] == [PROMPT + "q<|endoftext|>"]


@respx.mock
def test_wrong_dim_is_a_contract_violation_without_retry() -> None:
    route = respx.post(f"{BASE}/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})
    )
    with pytest.raises(ProviderContractViolation, match="dim"):
        _embedder().embed_passages(["a"])
    assert route.call_count == 1


@respx.mock
def test_nan_is_a_contract_violation() -> None:
    respx.post(f"{BASE}/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            content=b'{"data": [{"index": 0, "embedding": [NaN, 1.0, 0.0, 0.0]}]}',
            headers={"content-type": "application/json"},
        )
    )
    with pytest.raises(ProviderContractViolation, match="non-finite"):
        _embedder().embed_passages(["a"])


@respx.mock
def test_missing_items_are_a_contract_violation() -> None:
    respx.post(f"{BASE}/v1/embeddings").mock(return_value=httpx.Response(200, json={"data": []}))
    with pytest.raises(ProviderContractViolation):
        _embedder().embed_passages(["a"])


@respx.mock
def test_timeout_is_retried_then_provider_timeout() -> None:
    route = respx.post(f"{BASE}/v1/embeddings").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderTimeout):
        _embedder().embed_passages(["a"])
    assert route.call_count == FAST.attempts


@respx.mock
def test_connection_refused_is_provider_unavailable() -> None:
    route = respx.post(f"{BASE}/v1/embeddings").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ProviderUnavailable):
        _embedder().embed_queries(["q"])
    assert route.call_count == FAST.attempts


@respx.mock
def test_server_error_retries_then_recovers() -> None:
    route = respx.post(f"{BASE}/v1/embeddings").mock(
        side_effect=[httpx.Response(503), httpx.Response(500), _echo_embeddings]
    )
    mat = _embedder().embed_passages(["a"])
    assert route.call_count == 3 and mat.shape == (1, DIM)


@respx.mock
def test_client_error_is_not_retried() -> None:
    route = respx.post(f"{BASE}/v1/embeddings").mock(return_value=httpx.Response(400, text="bad"))
    with pytest.raises(ProviderContractViolation, match="HTTP 400"):
        _embedder().embed_passages(["a"])
    assert route.call_count == 1


@respx.mock
def test_token_count_uses_tokenize_with_special_tokens_and_caches() -> None:
    route = respx.post(f"{BASE}/tokenize").mock(
        return_value=httpx.Response(200, json={"tokens": [1, 2, 3]})
    )
    e = _embedder()
    assert e.token_count("бюджет") == 3
    assert e.token_count("бюджет") == 3
    assert route.call_count == 1
    sent = json.loads(route.calls[0].request.content)
    assert sent == {"content": "бюджет", "add_special": True}


def test_profile_carries_gguf_identity() -> None:
    o = OpenAICompatEmbedConfig(gguf_sha256="abc", quant="Q6_K")
    p = EmbeddingConfig(provider="openai_compat", openai_compat=o).profile()
    assert p.provider == "openai_compat" and p.pooling == "last" and p.dim == 2560
    assert p.key().endswith(":Q6_K:abc")


# ---- reranker ----


def _reranker(transform: str = "none") -> OpenAICompatReranker:  # raw scores unless asked
    o = OpenAICompatRerankConfig(base_url=RBASE, retry=FAST, score_transform=transform)
    return OpenAICompatReranker(
        RerankConfig(enabled=True, provider="openai_compat", openai_compat=o)
    )


@respx.mock
def test_rerank_maps_scores_back_to_input_order() -> None:
    route = respx.post(f"{RBASE}/v1/rerank").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 3.0},
                    {"index": 0, "relevance_score": 1.0},
                    {"index": 1, "relevance_score": -2.0},
                ]
            },
        )
    )
    assert _reranker().score("q", ["a", "b", "c"]) == [1.0, -2.0, 3.0]
    sent = json.loads(route.calls[0].request.content)
    assert sent["query"] == "q" and sent["documents"] == ["a", "b", "c"] and sent["top_n"] == 3


@respx.mock
def test_rerank_sigmoid_transform() -> None:
    respx.post(f"{RBASE}/v1/rerank").mock(
        return_value=httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.0}]})
    )
    assert _reranker("sigmoid").score("q", ["a"]) == [0.5]
    assert math.isclose(1 / (1 + math.exp(-2.0)), 0.8807970779778823)


@respx.mock
def test_rerank_incomplete_results_violate_the_contract() -> None:
    respx.post(f"{RBASE}/v1/rerank").mock(
        return_value=httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})
    )
    with pytest.raises(ProviderContractViolation, match="cover"):
        _reranker().score("q", ["a", "b"])


def test_rerank_empty_makes_no_request() -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{RBASE}/v1/rerank")
        assert _reranker().score("q", []) == []
        assert route.call_count == 0


@respx.mock
def test_rerank_unavailable() -> None:
    respx.post(f"{RBASE}/v1/rerank").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ProviderUnavailable):
        _reranker().score("q", ["a"])
