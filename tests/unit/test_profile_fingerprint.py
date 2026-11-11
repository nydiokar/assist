"""Profile fingerprint (S3 / A4, ADR-009): every identity field of the served embedder changes the
profile key AND the generation id; a mismatch with the pinned project profile refuses writes."""

from __future__ import annotations

from pathlib import Path

import pytest

from project_rag.core.fingerprint import generation_id, pipeline_fingerprint
from project_rag.domain.config import (
    ChunkingConfig,
    ClassificationConfig,
    Config,
    EmbeddingConfig,
    OpenAICompatEmbedConfig,
    ParsingConfig,
    with_overrides,
)
from project_rag.wiring import build
from tests.conftest import FIXTURE_CORPUS

BASE = OpenAICompatEmbedConfig(gguf_sha256="a" * 64)

CHANGES: dict[str, object] = {
    "model_id": "Qwen/Qwen3-Embedding-8B-GGUF",
    "revision": "0" * 40,
    "gguf_sha256": "b" * 64,
    "quant": "Q4_K_M",
    "pooling": "mean",
    "query_prompt_version": "qwen3_query_v2",
    "dim": 1024,
    "normalize": False,
}


def _gen(o: OpenAICompatEmbedConfig) -> tuple[str, str]:
    key = EmbeddingConfig(provider="openai_compat", openai_compat=o).profile().key()
    pipe = pipeline_fingerprint(ParsingConfig(), ChunkingConfig(), ClassificationConfig(), key)
    return key, generation_id("demo", ["h1a.md", "h2b.md"], pipe)


@pytest.mark.parametrize("field", sorted(CHANGES))
def test_any_profile_field_changes_key_and_generation(field: str) -> None:
    base_key, base_gen = _gen(BASE)
    key, gen = _gen(BASE.model_copy(update={field: CHANGES[field]}))
    assert key != base_key
    assert gen != base_gen


def test_provider_change_changes_generation() -> None:
    hash_key = EmbeddingConfig(provider="hash").profile().key()
    http_key, _ = _gen(BASE)
    pipe = [
        pipeline_fingerprint(ParsingConfig(), ChunkingConfig(), ClassificationConfig(), k)
        for k in (hash_key, http_key)
    ]
    assert generation_id("demo", ["x"], pipe[0]) != generation_id("demo", ["x"], pipe[1])


def test_same_profile_same_generation() -> None:
    assert _gen(BASE) == _gen(BASE.model_copy())


def test_mismatched_profile_refuses_writes(tmp_path: Path) -> None:
    common = {
        "lexical_store.path": str(tmp_path / "store.sqlite3"),
        "vector_store.persist_path": str(tmp_path / "vectors"),
    }
    first = build(with_overrides(Config(), common))
    report = first.ingest.ingest("demo", str(FIXTURE_CORPUS))
    gen = report.index_generation
    first.close()
    # Same store, different embedding profile (hash dim stands in for any served-model change).
    second = build(with_overrides(Config(), {**common, "embedding.hash_dim": 1024}))
    refused = second.ingest.ingest("demo", str(FIXTURE_CORPUS))
    assert refused.index_generation == ""
    assert any(w.startswith("REFUSED: embedding profile mismatch") for w in refused.warnings)
    assert second.catalog.active_generation("demo") == gen  # nothing was written
    second.close()
