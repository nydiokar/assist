"""Ingest service: idempotent ingestion + reconciliation (v1 §4.8), ports injected.

Flow: list files → compare hashes vs the active generation → parse changed/added (stage cache)
→ chunk (stage cache) → embed (embedding cache) → stage a NEW generation → publish → delete the
superseded generation's derivative records.

Guarantees:
- Re-run with no changes → no new embeddings, identical active membership (idempotent).
- Modified file → old chunks removed, new chunks present (no stale, no orphans on shrink).
- Deleted source → absent from the active searchable set.
- Model/config mismatch → writes refused (compatibility guard).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ..core.chunking import chunk_document
from ..core.corpus_diff import diff_corpus
from ..core.filesystem_evidence import build_filesystem_chunks
from ..core.fingerprint import (
    chunk_key,
    chunk_lineage_key,
    classify_doc,
    generation_id,
    parse_key,
    pipeline_fingerprint,
    source_id,
    text_hash,
    version_id,
)
from ..domain.config import ChunkingConfig, Config, ParsingConfig
from ..domain.models import (
    Chunk,
    ChunkArtifact,
    DocState,
    DocumentRecord,
    FileReport,
    IngestReport,
    ParseArtifact,
    ParsedDocument,
    StatusReport,
)
from ..ports.cache import ArtifactStore, EmbeddingCache
from ..ports.catalog import Catalog
from ..ports.corpus import CorpusReader
from ..ports.embedder import Embedder
from ..ports.lexical import LexicalIndex
from ..ports.parser import DocumentParser
from ..ports.vector import VectorIndex


class StageRunner:
    """Reusable, independently re-runnable parse/chunk stages over an ArtifactStore (A01 §C).

    An unchanged source is not re-parsed or re-chunked; a chunker-config change invalidates only
    the chunk stage; a parser-config change invalidates parse and downstream; an embedding-profile
    change re-embeds without touching either. Counts hits/misses for the ingest report.
    """

    def __init__(self, store: ArtifactStore, project_id: str) -> None:
        self._store = store
        self._project_id = project_id
        self.parse_hits = 0
        self.parse_misses = 0
        self.chunk_hits = 0
        self.chunk_misses = 0

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


class IngestService:
    def __init__(
        self,
        cfg: Config,
        *,
        corpus: CorpusReader,
        parser: DocumentParser,
        embedder: Embedder,
        vector: VectorIndex,
        lexical: LexicalIndex,
        catalog: Catalog,
        artifacts: ArtifactStore,
        open_embedding_cache: Callable[[], EmbeddingCache],
    ) -> None:
        self._cfg = cfg
        self._corpus = corpus
        self._parser = parser
        self._embedder = embedder
        self._vector = vector
        self._lexical = lexical
        self._catalog = catalog
        self._artifacts = artifacts
        self._open_embedding_cache = open_embedding_cache

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def ingest(self, project_id: str, root_path: str, *, dry_run: bool = False) -> IngestReport:
        profile = self._embedder.profile
        collection = self._cfg.vector_store.collection

        problems = self._catalog.check_compatibility(project_id, profile, collection)
        if problems:
            return IngestReport(
                project_id=project_id,
                index_generation="",
                embedding_profile_key=profile.key(),
                warnings=["REFUSED: " + p for p in problems],
            )

        if profile.dim != self._embedder.dim:  # defensive
            return IngestReport(
                project_id=project_id,
                index_generation="",
                warnings=["REFUSED: embedder dim != profile dim"],
            )

        discovered = self._corpus.list_files(root_path)
        pipeline = pipeline_fingerprint(
            self._cfg.parsing, self._cfg.chunking, self._cfg.classification
        )
        new_gen = generation_id(
            project_id, [d.content_hash + d.source_path for d in discovered], pipeline
        )
        prior_gen = self._catalog.active_generation(project_id)
        prior_hashes = self._catalog.prior_hashes(project_id, prior_gen) if prior_gen else {}

        files: list[FileReport] = []
        warnings: list[str] = []
        changes = diff_corpus(prior_hashes, discovered)
        counts: dict[str, int] = {
            "added": len(changes.added),
            "modified": len(changes.modified),
            "unchanged": len(changes.unchanged),
            "deleted": len(changes.deleted),
        }

        def report() -> IngestReport:
            return IngestReport(
                project_id=project_id,
                index_generation=new_gen,
                dry_run=dry_run,
                embedding_profile_key=profile.key(),
                files=files,
                warnings=warnings,
                **counts,
            )

        # Short-circuit true idempotency: identical corpus AND same generation already active.
        if prior_gen == new_gen and not dry_run:
            counts = {"added": 0, "modified": 0, "unchanged": len(discovered), "deleted": 0}
            warnings.append("no changes: active generation unchanged")
            for d in discovered:
                files.append(
                    FileReport(
                        source_path=d.source_path,
                        state=DocState.READY if d.supported else DocState.UNSUPPORTED,
                        source_id=source_id(d.source_path),
                        version_id=version_id(d.content_hash, self._cfg.parsing.backend),
                    )
                )
            return report()

        stages = StageRunner(self._artifacts, project_id)
        token_counter = self._embedder.token_count
        staged_chunks: list[Chunk] = []
        for d in discovered:
            sid = source_id(d.source_path)
            vid = version_id(d.content_hash, self._cfg.parsing.backend)

            if not d.supported:
                files.append(
                    FileReport(
                        source_path=d.source_path,
                        state=DocState.UNSUPPORTED,
                        source_id=sid,
                        version_id=vid,
                        diagnostics=[f"unsupported ext {d.ext}"],
                    )
                )
                continue

            parse_art = stages.get_or_parse(
                abs_path=d.abs_path,
                source_path=d.source_path,
                ext=d.ext,
                content_hash=d.content_hash,
                registry=self._parser,
                parser_cfg=self._cfg.parsing,
            )
            doc: ParsedDocument = parse_art.doc
            if doc.state not in {DocState.PARSED}:
                files.append(
                    FileReport(
                        source_path=d.source_path,
                        state=doc.state,
                        source_id=sid,
                        version_id=vid,
                        diagnostics=doc.diagnostics,
                    )
                )
                continue

            chunk_art = stages.get_or_chunk(
                parse_art=parse_art,
                project_id=project_id,
                source_id=sid,
                version_id=vid,
                source_type=d.ext.lstrip("."),
                chunk_cfg=self._cfg.chunking,
                token_counter=token_counter,
            )
            doc_class = classify_doc(
                d.source_path, tuple(self._cfg.classification.reference_prefixes)
            )
            chunks = [c.model_copy(update={"doc_class": doc_class}) for c in chunk_art.chunks]
            staged_chunks.extend(chunks)
            diags = list(doc.diagnostics)
            if any(c.truncated for c in chunks):  # A01 §F: silent truncation is a failed check
                diags.append(
                    f"TRUNCATION: {sum(c.truncated for c in chunks)} chunk(s) exceeded "
                    f"max_tokens={self._cfg.chunking.max_tokens}"
                )
            files.append(
                FileReport(
                    source_path=d.source_path,
                    state=DocState.READY,
                    source_id=sid,
                    version_id=vid,
                    n_blocks=len(doc.blocks),
                    n_chunks=len(chunks),
                    diagnostics=diags,
                )
            )

        # Filesystem-presence evidence for unparseable files (DECISIONS 2026-10-28): synthesized
        # from the discovery walk; doc_class=filesystem keeps it filterable and unmistakable.
        if self._cfg.classification.filesystem_evidence:
            fs_chunks = build_filesystem_chunks(project_id, discovered)
            staged_chunks.extend(fs_chunks)
            if fs_chunks:
                warnings.append(f"filesystem_evidence: {len(fs_chunks)} presence chunk(s)")

        if dry_run:
            warnings.append("dry run: no writes performed")
            return report()

        # Stage: embed + write the new generation.
        if staged_chunks:
            embeddings = self._embed_cached(staged_chunks)
            self._vector.add(new_gen, staged_chunks, embeddings)
            self._lexical.add_chunks(new_gen, staged_chunks)
            self._catalog.record_chunks(
                project_id,
                new_gen,
                [(c.chunk_id, c.source_id, c.version_id) for c in staged_chunks],
            )

        for fr in files:
            content_hash = next(
                (d.content_hash for d in discovered if d.source_path == fr.source_path), ""
            )
            self._catalog.upsert_document(
                DocumentRecord(
                    project_id=project_id,
                    source_id=fr.source_id,
                    source_path=fr.source_path,
                    content_hash=content_hash,
                    version_id=fr.version_id or "",
                    state=fr.state,
                    generation=new_gen,
                    n_chunks=fr.n_chunks,
                    diagnostics="; ".join(fr.diagnostics),
                )
            )

        # Publish the new active generation, then remove superseded derivatives.
        self._catalog.set_project_profile(project_id, profile, collection)
        self._catalog.publish_generation(project_id, new_gen)
        if prior_gen and prior_gen != new_gen:
            old_ids = self._catalog.chunk_ids_for_generation(project_id, prior_gen)
            self._vector.delete_ids(old_ids)
            self._lexical.delete_generation(project_id, prior_gen)
            self._catalog.delete_generation(project_id, prior_gen)

        warnings.append(f"stage_cache={stages.stats()}")
        return report()

    def _embed_cached(self, chunks: list[Chunk], *, batch: int = 64) -> npt.NDArray[np.float32]:
        """Embed only chunks whose (text, profile) is not cached; commit each batch (resumable)."""
        profile_key = self._embedder.profile.key()
        dim = self._embedder.dim
        cache = self._open_embedding_cache()
        try:
            hashes = [text_hash(c.embed_text) for c in chunks]
            have = cache.get_many(profile_key, sorted(set(hashes)), dim)
            todo: list[tuple[str, str]] = []
            seen: set[str] = set()
            for h, c in zip(hashes, chunks, strict=True):
                if h in have or h in seen:
                    continue
                seen.add(h)
                todo.append((h, c.embed_text))
            for i in range(0, len(todo), batch):
                part = todo[i : i + batch]
                vecs = self._embedder.embed_passages([t for _, t in part])
                new = {h: v for (h, _), v in zip(part, vecs, strict=True)}
                cache.put_many(profile_key, new)
                have.update(new)
            if not hashes:
                return np.zeros((0, dim), dtype=np.float32)
            return np.stack([have[h] for h in hashes])
        finally:
            cache.close()

    def status(self, project_id: str) -> StatusReport:
        gen = self._catalog.active_generation(project_id)
        if gen is None:
            return StatusReport(
                project_id=project_id,
                index_generation=None,
                n_documents=0,
                n_chunks=0,
                healthy=False,
                warnings=["no active generation"],
            )
        docs = self._catalog.document_states(project_id, gen)
        states: dict[str, int] = {}
        n_chunks = 0
        for r in docs:
            states[r.state] = states.get(r.state, 0) + 1
            n_chunks += r.n_chunks
        return StatusReport(
            project_id=project_id,
            index_generation=gen,
            n_documents=len(docs),
            n_chunks=n_chunks,
            states=states,
            embedding_profile_key=self._embedder.profile.key(),
            collection=self._cfg.vector_store.collection,
            healthy=True,
        )
