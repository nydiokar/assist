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

from collections.abc import Callable

from .config import ChunkingConfig, ParsingConfig
from .core.chunking import chunk_document
from .core.fingerprint import chunk_key, chunk_lineage_key, parse_key
from .domain.models import ChunkArtifact, ParseArtifact
from .ports.cache import ArtifactStore
from .ports.parser import DocumentParser


class StageCache:
    """Parse artifacts are shared across projects (keyed by content hash + parser config only),
    because identical bytes parse identically regardless of project. Chunk artifacts are
    per-project because chunk IDENTITY carries project/source/version lineage and distinct
    provenance/access scope (A01 §C): the raw parsed text may be shared, but index membership
    and addresses must stay separate.
    """

    def __init__(self, store: ArtifactStore, project_id: str) -> None:
        self._store = store
        self._project_id = project_id
        self.parse_hits = 0
        self.parse_misses = 0
        self.chunk_hits = 0
        self.chunk_misses = 0

    # ---- [1] PARSE ----
    def get_or_parse(
        self,
        *,
        abs_path: str,
        source_path: str,
        ext: str,
        content_hash: str,
        registry: DocumentParser,
        parser_cfg: ParsingConfig,
    ) -> ParseArtifact:
        key = parse_key(content_hash, parser_cfg)
        cached = self._store.get_parse(key)
        if cached is not None:
            self.parse_hits += 1
            return cached
        self.parse_misses += 1
        doc = registry.parse(abs_path, source_path, ext)
        art = ParseArtifact(
            artifact_id=key, source_path=source_path, source_content_hash=content_hash, doc=doc
        )
        self._store.put_parse(art)
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
        token_counter: Callable[[str], int] | None = None,
    ) -> ChunkArtifact:
        key = chunk_key(parse_art.artifact_id, chunk_cfg)
        lineage = chunk_lineage_key(key, project_id, source_id, version_id)
        cached = self._store.get_chunk(self._project_id, lineage)
        if cached is not None:
            self.chunk_hits += 1
            return cached
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
        self._store.put_chunk(self._project_id, art)
        return art

    def stats(self) -> dict[str, int]:
        return {
            "parse_hits": self.parse_hits,
            "parse_misses": self.parse_misses,
            "chunk_hits": self.chunk_hits,
            "chunk_misses": self.chunk_misses,
        }
