"""S6 exit check against the real llama-server pair (`live`): stopping the embed and/or rerank
container gives flagged results (`degraded`), never an empty list for a server fault, and
/healthz reflects each state. Needs Docker + deploy/dev/compose.models.yaml up.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from project_rag.domain.config import RetryPolicy, with_overrides
from project_rag.runtime import Runtime, build_runtime
from project_rag.wiring import load_config
from tests.conftest import FIXTURE_CORPUS

pytestmark = pytest.mark.live
REPO = Path(__file__).resolve().parents[2]
COMPOSE = ["docker", "compose", "-f", str(REPO / "deploy" / "dev" / "compose.models.yaml")]
FAST = RetryPolicy(attempts=1, backoff_s=0.0, max_backoff_s=0.0).model_dump()


def _compose(*args: str) -> None:
    subprocess.run([*COMPOSE, *args], check=True, capture_output=True, timeout=180)


def _healthy(url: str) -> bool:
    try:
        return httpx.get(url, timeout=3).is_success
    except httpx.HTTPError:
        return False


def _wait(url: str) -> None:
    deadline = time.monotonic() + 180
    while not _healthy(url):
        assert time.monotonic() < deadline, f"{url} did not come back"
        time.sleep(2)


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Runtime]:
    if shutil.which("docker") is None:
        pytest.skip("docker not available")
    base = load_config(REPO / "configs" / "dev_llama_server.yaml")
    if not (_healthy("http://127.0.0.1:8081/health") and _healthy("http://127.0.0.1:8082/health")):
        pytest.skip("llama-server pair not running")
    root = tmp_path / "corpus"
    shutil.copytree(FIXTURE_CORPUS, root / "Demo")
    cfg = with_overrides(
        base,
        {
            "lexical_store.path": str(tmp_path / "state" / "store.sqlite3"),
            "vector_store.provider": "sqlite_vec",
            "vector_store.persist_path": "",
            "service.corpus_root": str(root),
            "embedding.openai_compat.retry": FAST,
            "retrieval.rerank.openai_compat.retry": FAST,
        },
    )
    rt = build_runtime(cfg, token="t")
    rt.worker.trigger()
    rt.worker.join(600)
    assert rt.worker.last_error is None, rt.worker.last_error
    rt.executor.submit_sync(rt.api.sync_registry)
    yield rt
    rt.close()


def _search(rt: Runtime) -> tuple[list[str], int]:
    resp = rt.executor.submit_sync(
        lambda: rt.services.search.search_project(
            "demo", "What grant number is the project under?", top_k=5
        )
    )
    return resp.degraded, len(resp.results)


def _health(rt: Runtime) -> dict[str, str]:
    return rt.executor.submit_sync(rt.api.health)


def test_rerank_down(runtime: Runtime) -> None:
    assert _search(runtime) == ([], 5) and _health(runtime)["rerank"] == "ok"
    _compose("stop", "rerank")
    try:
        degraded, n = _search(runtime)
        assert degraded == ["rerank"] and n > 0
        assert _health(runtime)["rerank"] == "unreachable"
    finally:
        _compose("start", "rerank")
        _wait("http://127.0.0.1:8082/health")
    assert _search(runtime) == ([], 5)


def test_embed_down_and_both(runtime: Runtime) -> None:
    _compose("stop", "embed")
    try:
        degraded, n = _search(runtime)
        assert degraded == ["dense"] and n > 0
        assert _health(runtime)["embed"] == "unreachable"
        _compose("stop", "rerank")
        try:
            degraded, n = _search(runtime)
            assert degraded == ["dense", "rerank"] and n > 0  # lexical only, flagged
            h = _health(runtime)
            assert h["embed"] == h["rerank"] == "unreachable" and h["store"] == "ok"
        finally:
            _compose("start", "rerank")
            _wait("http://127.0.0.1:8082/health")
    finally:
        _compose("start", "embed")
        _wait("http://127.0.0.1:8081/health")
    assert _search(runtime) == ([], 5)
