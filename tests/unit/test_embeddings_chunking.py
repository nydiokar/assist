"""Embeddings prefix/dim/determinism and chunk-boundary tests (v1 §6.3)."""

from __future__ import annotations

import numpy as np

from project_rag.chunking import chunk_document
from project_rag.config import ChunkingConfig, EmbeddingConfig
from project_rag.embeddings import HashEmbedder
from project_rag.schema import Block, BlockType, DocState, ParsedDocument, SourceLocation


def test_hash_embedder_dim_and_normalized():
    emb = HashEmbedder(EmbeddingConfig(provider="hash", hash_dim=384))
    v = emb.embed_passages(["hello world"])
    assert v.shape == (1, 384)
    assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-5


def test_hash_embedder_deterministic():
    emb = HashEmbedder(EmbeddingConfig(provider="hash"))
    a = emb.embed_queries(["indicator target 120"])
    b = emb.embed_queries(["indicator target 120"])
    assert np.allclose(a, b)


def test_query_passage_prefixes_differ():
    emb = HashEmbedder(EmbeddingConfig(provider="hash"))
    q = emb.embed_queries(["same text"])
    p = emb.embed_passages(["same text"])
    # asymmetric prefixes must change the vector
    assert not np.allclose(q, p)


def test_profile_key_changes_with_provider():
    hash_key = EmbeddingConfig(provider="hash").profile().key()
    e5_key = EmbeddingConfig(provider="e5").profile().key()
    assert hash_key != e5_key


def _doc() -> ParsedDocument:
    return ParsedDocument(
        source_path="d.md", state=DocState.PARSED,
        blocks=[
            Block(block_type=BlockType.HEADING, text="Project indicators"),
            Block(block_type=BlockType.PARAGRAPH,
                  text="Indicator 1: people trained. Target: 120 people.",
                  location=SourceLocation(heading="Project indicators")),
            Block(block_type=BlockType.TABLE_ROW, text="indicator=1, target=120, unit=people",
                  location=SourceLocation(sheet="Indicators", row=2)),
        ],
    )


def test_chunk_ids_deterministic_and_prefixed():
    kw = dict(project_id="p", source_id="s", version_id="v", source_type="md",
              cfg=ChunkingConfig())
    c1 = chunk_document(_doc(), **kw)
    c2 = chunk_document(_doc(), **kw)
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]
    assert all(c.chunk_id.startswith("ck_") for c in c1)


def test_target_and_unit_stay_in_same_span():
    chunks = chunk_document(_doc(), project_id="p", source_id="s", version_id="v",
                            source_type="md", cfg=ChunkingConfig())
    text_chunk = [c for c in chunks if "Target: 120" in c.citation_text][0]
    # the heading context travels with the chunk for embedding
    assert "Project indicators" in text_chunk.embed_text
    table_chunk = [c for c in chunks if c.block_type == BlockType.TABLE][0]
    assert "target=120" in table_chunk.citation_text and "people" in table_chunk.citation_text
