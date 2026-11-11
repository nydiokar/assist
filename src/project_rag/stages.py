"""Reusable, independently-rerunnable pipeline stages with fingerprinted caches (A01 §C).

Each stage persists its artifact keyed by a fingerprint of its inputs so that:
  - an unchanged source is NOT reparsed/rechunked (cache hit),
  - changing the chunker config invalidates only [2] CHUNK and downstream (not [1] PARSE),
  - changing the parser config invalidates [1] PARSE and downstream,
  - changing the embedding profile re-embeds without touching [1]/[2].

Artifacts live under local_data/cache/<project_id>/{parse,chunk}/<key>.json and are plain JSON
(Pydantic round-trip). Stage [3] EMBED is keyed in the vector store by chunk + embedding_profile;
stages.py owns [1] PARSE and [2] CHUNK, the two stages that are expensive to recompute.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel

from .chunking import CHUNKER_VERSION, chunk_document
from .config import ChunkingConfig, ParsingConfig
from .parsers import ParserRegistry
from .schema import Chunk, ParsedDocument

PARSER_VERSION = "native_v1"


def _fp(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


def parse_key(source_content_hash: str, parser_cfg: ParsingConfig) -> str:
    return _fp("parse", source_content_hash, PARSER_VERSION, parser_cfg.model_dump_json())


def chunk_key(parsed_artifact_id: str, chunk_cfg: ChunkingConfig) -> str:
    return _fp("chunk", parsed_artifact_id, CHUNKER_VERSION, chunk_cfg.model_dump_json())


class ParseArtifact(BaseModel):
    artifact_id: str  # == parse_key (identity for downstream chunk key)
    source_path: str
    source_content_hash: str
    doc: ParsedDocument


class ChunkArtifact(BaseModel):
    artifact_id: str  # == chunk_key
    parsed_artifact_id: str
    chunks: list[Chunk]


class StageCache:
    """Parse artifacts are shared across projects (keyed by content hash + parser config only),
    because identical bytes parse identically regardless of project. Chunk artifacts are
    per-project because chunk IDENTITY carries project/source/version lineage and distinct
    provenance/access scope (A01 §C): the raw parsed text may be shared, but index membership
    and addresses must stay separate.
    """

    def __init__(self, root: str | Path, project_id: str) -> None:
        self._root = Path(root)
        self._parse_dir = self._root / "_parse_shared"  # cross-project
        self._chunk_dir = self._root / project_id / "chunk"
        self._parse_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_dir.mkdir(parents=True, exist_ok=True)
        self.parse_hits = 0
        self.parse_misses = 0
        self.chunk_hits = 0
        self.chunk_misses = 0

    def _parse_path(self, key: str) -> Path:
        return self._parse_dir / f"{key}.json"

    def _chunk_path(self, key: str) -> Path:
        return self._chunk_dir / f"{key}.json"

    # ---- [1] PARSE ----
    def get_or_parse(
        self,
        *,
        abs_path: str,
        source_path: str,
        ext: str,
        content_hash: str,
        registry: ParserRegistry,
        parser_cfg: ParsingConfig,
    ) -> ParseArtifact:
        key = parse_key(content_hash, parser_cfg)
        p = self._parse_path(key)
        if p.is_file():
            self.parse_hits += 1
            return ParseArtifact.model_validate_json(p.read_text(encoding="utf-8"))
        self.parse_misses += 1
        doc = registry.parse(abs_path, source_path, ext)
        art = ParseArtifact(
            artifact_id=key, source_path=source_path, source_content_hash=content_hash, doc=doc
        )
        p.write_text(art.model_dump_json(), encoding="utf-8")
        return art

    # ---- [2] CHUNK ----
    def get_or_chunk(
        self,
        *,
        parse_art: ParseArtifact,
        project_id: str,
        source_id: str,
        version_id: str,
        source_type: str,
        chunk_cfg: ChunkingConfig,
        token_counter=None,
    ) -> ChunkArtifact:
        key = chunk_key(parse_art.artifact_id, chunk_cfg)
        # Chunk identity also depends on project/source/version lineage; fold into filename.
        lineage = _fp(key, project_id, source_id, version_id)
        p = self._chunk_path(lineage)
        if p.is_file():
            self.chunk_hits += 1
            return ChunkArtifact.model_validate_json(p.read_text(encoding="utf-8"))
        self.chunk_misses += 1
        chunks = chunk_document(
            parse_art.doc,
            project_id=project_id,
            source_id=source_id,
            version_id=version_id,
            source_type=source_type,
            cfg=chunk_cfg,
            token_counter=token_counter,
        )
        art = ChunkArtifact(
            artifact_id=lineage, parsed_artifact_id=parse_art.artifact_id, chunks=chunks
        )
        p.write_text(art.model_dump_json(), encoding="utf-8")
        return art

    def stats(self) -> dict:
        return {
            "parse_hits": self.parse_hits,
            "parse_misses": self.parse_misses,
            "chunk_hits": self.chunk_hits,
            "chunk_misses": self.chunk_misses,
        }
