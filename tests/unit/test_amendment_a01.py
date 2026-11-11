"""Amendment A01 regression cases (§C stages, §D profile, §F chunking)."""

from __future__ import annotations

from pathlib import Path

from project_rag.chunking import chunk_document
from project_rag.config import (
    ChunkingConfig,
    Config,
    EmbeddingConfig,
    ParsingConfig,
    with_overrides,
)
from project_rag.parsers import ParserRegistry
from project_rag.schema import Block, BlockType, DocState, ParsedDocument
from project_rag.stages import StageCache

# ---- §D profile contract ----


def test_qwen3_profile_has_prompt_not_prefix():
    prof = EmbeddingConfig(provider="qwen3").profile()
    assert prof.provider == "qwen3"
    assert prof.dim == 2560
    assert prof.pooling == "last_token"
    assert prof.query_prompt_name == "query"
    # A01: no E5 prefix leakage into the Qwen3 profile
    assert prof.query_prefix == "" and prof.passage_prefix == ""
    assert prof.device == "cuda"


def test_profile_key_distinguishes_providers_and_dims():
    q = EmbeddingConfig(provider="qwen3").profile().key()
    e5 = EmbeddingConfig(provider="e5").profile().key()
    h = EmbeddingConfig(provider="hash").profile().key()
    assert len({q, e5, h}) == 3


def test_release_guard_rejects_unpinned_qwen3_revision():
    cfg = with_overrides(
        Config(), {"embedding.provider": "qwen3", "vector_store.provider": "chroma"}
    )
    problems = cfg.assert_release_safe()
    assert any("revision not pinned" in p for p in problems)
    cfg = with_overrides(cfg, {"embedding.qwen3.revision": "abc123deadbeef"})
    assert cfg.assert_release_safe() == []


# ---- §C reusable stages ----


def _md(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text(
        "# Head\n\nalpha beta gamma. delta epsilon zeta.\n", encoding="utf-8"
    )
    return root


def test_changing_chunk_config_reuses_parse(tmp_path):
    root = _md(tmp_path)
    reg = ParserRegistry(ParsingConfig())
    cache = StageCache(tmp_path / "cache", "p")
    content_hash = "h1"
    pa = cache.get_or_parse(
        abs_path=str(root / "a.md"),
        source_path="a.md",
        ext=".md",
        content_hash=content_hash,
        registry=reg,
        parser_cfg=ParsingConfig(),
    )
    cache.get_or_chunk(
        parse_art=pa,
        project_id="p",
        source_id="s",
        version_id="v",
        source_type="md",
        chunk_cfg=ChunkingConfig(target_tokens=500),
    )
    assert cache.parse_misses == 1 and cache.chunk_misses == 1

    # New cache instance (simulates a later run) with a DIFFERENT chunk config.
    cache2 = StageCache(tmp_path / "cache", "p")
    pa2 = cache2.get_or_parse(
        abs_path=str(root / "a.md"),
        source_path="a.md",
        ext=".md",
        content_hash=content_hash,
        registry=reg,
        parser_cfg=ParsingConfig(),
    )
    cache2.get_or_chunk(
        parse_art=pa2,
        project_id="p",
        source_id="s",
        version_id="v",
        source_type="md",
        chunk_cfg=ChunkingConfig(target_tokens=350),
    )
    assert cache2.parse_hits == 1  # parse reused (A01 §C)
    assert cache2.chunk_misses == 1  # re-chunked due to config change


def test_unchanged_source_full_cache_hit(tmp_path):
    root = _md(tmp_path)
    reg = ParserRegistry(ParsingConfig())
    cfg = ChunkingConfig()
    c1 = StageCache(tmp_path / "cache", "p")
    pa = c1.get_or_parse(
        abs_path=str(root / "a.md"),
        source_path="a.md",
        ext=".md",
        content_hash="h1",
        registry=reg,
        parser_cfg=ParsingConfig(),
    )
    c1.get_or_chunk(
        parse_art=pa, project_id="p", source_id="s", version_id="v", source_type="md", chunk_cfg=cfg
    )
    c2 = StageCache(tmp_path / "cache", "p")
    pa2 = c2.get_or_parse(
        abs_path=str(root / "a.md"),
        source_path="a.md",
        ext=".md",
        content_hash="h1",
        registry=reg,
        parser_cfg=ParsingConfig(),
    )
    c2.get_or_chunk(
        parse_art=pa2,
        project_id="p",
        source_id="s",
        version_id="v",
        source_type="md",
        chunk_cfg=cfg,
    )
    assert c2.parse_hits == 1 and c2.chunk_hits == 1
    assert c2.parse_misses == 0 and c2.chunk_misses == 0


# ---- §F chunking token counting / truncation ----


def test_oversized_chunk_marked_truncated():
    big = " ".join(["word"] * 5000)
    doc = ParsedDocument(
        source_path="d.md",
        state=DocState.PARSED,
        blocks=[Block(block_type=BlockType.PARAGRAPH, text=big)],
    )
    chunks = chunk_document(
        doc,
        project_id="p",
        source_id="s",
        version_id="v",
        source_type="md",
        cfg=ChunkingConfig(max_tokens=100),
        token_counter=lambda t: len(t.split()),
    )
    assert any(c.truncated for c in chunks)
    tc = next(c for c in chunks if c.truncated)
    assert tc.token_count <= 100
    # citation_text keeps the verbatim original even when embed_text is capped
    assert len(tc.citation_text.split()) >= len(tc.embed_text.split())


def test_retrieval_text_alias_matches_embed_text():
    doc = ParsedDocument(
        source_path="d.md",
        state=DocState.PARSED,
        blocks=[
            Block(block_type=BlockType.HEADING, text="H"),
            Block(block_type=BlockType.PARAGRAPH, text="body text"),
        ],
    )
    chunks = chunk_document(
        doc, project_id="p", source_id="s", version_id="v", source_type="md", cfg=ChunkingConfig()
    )
    assert chunks[0].retrieval_text == chunks[0].embed_text
