"""Provider faults against the real pinned llama-server (S3 exit check, `live`).

A stopped container must surface as `ProviderUnavailable` (connection refused), never as an
empty or partial result; restarting it restores service. Needs Docker and the compose pair from
deploy/dev/compose.models.yaml (`make models-up`). The unit-level fault matrix (timeout, refused,
5xx, bad dim, NaN) runs offline in tests/unit/test_openai_compat_adapters.py.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from project_rag.adapters.embed_openai_compat import OpenAICompatEmbedder
from project_rag.adapters.rerank_openai_compat import OpenAICompatReranker
from project_rag.domain.config import RetryPolicy, with_overrides
from project_rag.domain.errors import ProviderUnavailable
from project_rag.wiring import load_config

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[2]
COMPOSE = ["docker", "compose", "-f", str(REPO / "deploy" / "dev" / "compose.models.yaml")]


def _compose(*args: str) -> None:
    subprocess.run([*COMPOSE, *args], check=True, capture_output=True, timeout=180)


def _wait_healthy(probe: object, seconds: float = 180.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if probe.healthy():  # type: ignore[attr-defined]
            return
        time.sleep(2)
    raise AssertionError("server did not come back")


@pytest.fixture
def cfg():
    if shutil.which("docker") is None:
        pytest.skip("docker not available")
    base = load_config(REPO / "configs" / "dev_llama_server.yaml")
    fast = RetryPolicy(attempts=2, backoff_s=0.1, max_backoff_s=0.2).model_dump()
    return with_overrides(
        base,
        {"embedding.openai_compat.retry": fast, "retrieval.rerank.openai_compat.retry": fast},
    )


def test_stopped_embed_container_is_provider_unavailable(cfg) -> None:
    e = OpenAICompatEmbedder(cfg.embedding)
    if not e.healthy():
        pytest.skip("embed server not running")
    assert e.embed_queries(["бюджет"]).shape == (1, 2560)
    _compose("stop", "embed")
    try:
        with pytest.raises(ProviderUnavailable):
            e.embed_queries(["бюджет"])
    finally:
        _compose("start", "embed")
        _wait_healthy(e)
    assert e.embed_queries(["бюджет"]).shape == (1, 2560)


def test_stopped_rerank_container_is_provider_unavailable(cfg) -> None:
    r = OpenAICompatReranker(cfg.retrieval.rerank)
    if not r.healthy():
        pytest.skip("rerank server not running")
    assert len(r.score("budget", ["a", "b"])) == 2
    _compose("stop", "rerank")
    try:
        with pytest.raises(ProviderUnavailable):
            r.score("budget", ["a", "b"])
    finally:
        _compose("start", "rerank")
        _wait_healthy(r)
    assert len(r.score("budget", ["a", "b"])) == 2
