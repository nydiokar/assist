"""S8 exit check: the service refuses to start with the hash embedder, a missing GGUF, or a GGUF
whose sha256 differs from the pinned one. Also the layered config loader (spec §6.7)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from project_rag.adapters.config_yaml import load_layered
from project_rag.domain.config import Config, with_overrides
from project_rag.runtime import build_runtime


def _http_config(tmp: Path, sha: str) -> Config:
    return with_overrides(
        Config(),
        {
            "embedding.provider": "openai_compat",
            "embedding.openai_compat.gguf_file": "embed.gguf",
            "embedding.openai_compat.gguf_sha256": sha,
            "vector_store.provider": "sqlite_vec",
            "vector_store.persist_path": "",
            "lexical_store.path": str(tmp / "s.sqlite3"),
            "service.models_dir": str(tmp / "models"),
        },
    )


@pytest.fixture(autouse=True)
def _no_dev_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASSIST_ALLOW_DEV_PROFILE", raising=False)


def test_refuses_hash_embedder(tmp_path: Path) -> None:
    cfg = with_overrides(Config(), {"lexical_store.path": str(tmp_path / "s.sqlite3")})
    with pytest.raises(SystemExit, match="embedding.provider=hash is test-only"):
        build_runtime(cfg, token="t")


def test_refuses_missing_gguf(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="model file missing"):
        build_runtime(_http_config(tmp_path, "a" * 64), token="t")


def test_refuses_mismatched_gguf_sha(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "embed.gguf").write_bytes(b"not the pinned weights")
    with pytest.raises(SystemExit, match="sha256 mismatch for embed.gguf"):
        build_runtime(_http_config(tmp_path, "b" * 64), token="t")


def test_matching_gguf_passes_the_model_check(tmp_path: Path) -> None:
    from project_rag.adapters.model_check import gguf_problems

    (tmp_path / "models").mkdir()
    payload = b"pinned weights"
    (tmp_path / "models" / "embed.gguf").write_bytes(payload)
    cfg = _http_config(tmp_path, hashlib.sha256(payload).hexdigest())
    assert gguf_problems(cfg, tmp_path / "models") == []


def test_refuses_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSIST_ALLOW_DEV_PROFILE", "1")
    monkeypatch.delenv("ASSIST_MCP_TOKEN", raising=False)
    cfg = with_overrides(Config(), {"lexical_store.path": str(tmp_path / "s.sqlite3")})
    with pytest.raises(SystemExit, match="ASSIST_MCP_TOKEN is not set"):
        build_runtime(cfg)


def test_layered_config_overlay_and_env_secret(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "retrieval:\n  top_k: 8\n  rerank:\n    enabled: true\n    top_n: 12\n", encoding="utf-8"
    )
    overlay = tmp_path / "deployment.yaml"
    overlay.write_text("retrieval:\n  rerank:\n    top_n: 6\n", encoding="utf-8")
    cfg = load_layered(base, overlay, env={"ASSIST_HOSTED_EMBED_API_KEY": "sk-test"})
    assert cfg.retrieval.top_k == 8 and cfg.retrieval.rerank.enabled  # kept from base
    assert cfg.retrieval.rerank.top_n == 6  # overlay wins, deep merge
    assert cfg.embedding.openai_compat.api_key == "sk-test"  # from the environment
    overlay.write_text("embedding:\n  openai_compat:\n    api_key: leaked\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must come from the environment"):
        load_layered(base, overlay, env={})
