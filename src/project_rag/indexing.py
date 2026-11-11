"""Idempotent ingestion orchestration + reconciliation (v1 §4.8).

Flow: discover → compare hashes vs active generation → parse changed/added → chunk → embed →
stage into a NEW generation → publish → delete superseded generation's derivative records.

Guarantees:
- Re-run with no changes → no new embeddings, identical active membership (idempotent).
- Modified file → old chunks removed, new chunks present (no stale, no orphans on shrink).
- Deleted source → absent from active searchable set.
- Model/config mismatch → writes refused (compatibility guard).
"""

from __future__ import annotations

import hashlib

import numpy as np

from .chunking import CHUNKER_VERSION
from .config import Config
from .discovery import discover
from .embedding_cache import EmbeddingCache, text_hash
from .embeddings import EmbedderProtocol
from .lexical_store import LexicalStore
from .manifest import Manifest
from .parsers import ParserRegistry
from .schema import Chunk, DocState, FileReport, IngestReport, ParsedDocument
from .stages import StageCache
from .vector_store import VectorStoreProtocol


def _source_id(source_path: str) -> str:
    return "src_" + hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:20]


def _version_id(content_hash: str, parser_backend: str) -> str:
    raw = f"{content_hash}|{parser_backend}|{CHUNKER_VERSION}"
    return "v_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _generation_id(project_id: str, hashes: list[str], pipeline: str) -> str:
    """Identity of the whole derived index: sources AND the pipeline that produced it.
    A chunker/parser/config change must yield a new generation (ADR-009), not 'no changes'."""
    raw = project_id + "|" + pipeline + "|" + "|".join(sorted(hashes))
    return "g_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class IndexingService:
    def __init__(
        self,
        cfg: Config,
        embedder: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
        lexical_store: LexicalStore,
        manifest: Manifest,
        registry: ParserRegistry,
        cache_root: str = "local_data/cache",
    ) -> None:
        self._cfg = cfg
        self._embedder = embedder
        self._vs = vector_store
        self._lx = lexical_store
        self._manifest = manifest
        self._registry = registry
        self._cache_root = cache_root

    def ingest(self, project_id: str, root_path: str, *, dry_run: bool = False) -> IngestReport:
        profile = self._embedder.profile
        collection = self._cfg.vector_store.collection

        problems = self._manifest.check_compatibility(project_id, profile, collection)
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

        discovered = discover(root_path)
        pipeline = "|".join(
            [
                CHUNKER_VERSION,
                self._cfg.parsing.backend,
                self._cfg.chunking.model_dump_json(),
                self._cfg.classification.model_dump_json(),
            ]
        )
        new_gen = _generation_id(
            project_id, [d.content_hash + d.source_path for d in discovered], pipeline
        )
        prior_gen = self._manifest.active_generation(project_id)
        prior_hashes = self._manifest.prior_hashes(project_id, prior_gen) if prior_gen else {}

        files: list[FileReport] = []
        warnings: list[str] = []
        counts = {"added": 0, "modified": 0, "unchanged": 0, "deleted": 0}

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
            counts["unchanged"] = len(discovered)
            warnings.append("no changes: active generation unchanged")
            for d in discovered:
                files.append(
                    FileReport(
                        source_path=d.source_path,
                        state=DocState.READY if d.supported else DocState.UNSUPPORTED,
                        source_id=_source_id(d.source_path),
                        version_id=_version_id(d.content_hash, self._cfg.parsing.backend),
                    )
                )
            return report()

        cache = StageCache(self._cache_root, project_id)
        token_counter = self._embedder.token_count
        staged_chunks: list[Chunk] = []
        for d in discovered:
            sid = _source_id(d.source_path)
            vid = _version_id(d.content_hash, self._cfg.parsing.backend)
            prior = prior_hashes.get(d.source_path)
            changed = prior is None or prior[0] != d.content_hash
            if prior is None:
                counts["added"] += 1
            elif changed:
                counts["modified"] += 1
            else:
                counts["unchanged"] += 1

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

            parse_art = cache.get_or_parse(
                abs_path=d.abs_path,
                source_path=d.source_path,
                ext=d.ext,
                content_hash=d.content_hash,
                registry=self._registry,
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

            chunk_art = cache.get_or_chunk(
                parse_art=parse_art,
                project_id=project_id,
                source_id=sid,
                version_id=vid,
                source_type=d.ext.lstrip("."),
                chunk_cfg=self._cfg.chunking,
                token_counter=token_counter,
            )
            prefixes = tuple(self._cfg.classification.reference_prefixes)
            doc_class = (
                "reference" if prefixes and d.source_path.startswith(prefixes) else "project_record"
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

        # Filesystem-presence evidence for unparseable files (DECISIONS 2026-10-28). Synthesized
        # from the discovery walk (no parse/chunk cache — it is cheap and derived purely from the
        # file inventory). doc_class=filesystem keeps it filterable and unmistakable for content.
        if self._cfg.classification.filesystem_evidence:
            from .filesystem import build_filesystem_chunks

            fs_chunks = build_filesystem_chunks(project_id, discovered)
            staged_chunks.extend(fs_chunks)
            if fs_chunks:
                warnings.append(f"filesystem_evidence: {len(fs_chunks)} presence chunk(s)")

        deleted = set(prior_hashes) - {d.source_path for d in discovered}
        counts["deleted"] = len(deleted)

        if dry_run:
            warnings.append("dry run: no writes performed")
            return report()

        # Stage: embed + write new generation.
        if staged_chunks:
            embeddings = self._embed_cached(staged_chunks, cache_root=self._cache_root)
            self._vs.add(new_gen, staged_chunks, embeddings)
            self._lx.add_chunks(new_gen, staged_chunks)
            self._manifest.record_chunks(
                project_id,
                new_gen,
                [(c.chunk_id, c.source_id, c.version_id) for c in staged_chunks],
            )

        for fr in files:
            content_hash = next(
                (d.content_hash for d in discovered if d.source_path == fr.source_path), ""
            )
            self._manifest.upsert_document(
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

        # Publish new active generation, then remove superseded derivatives.
        self._manifest.set_project_profile(project_id, profile, collection)
        self._manifest.publish_generation(project_id, new_gen)
        if prior_gen and prior_gen != new_gen:
            old_ids = self._manifest.chunk_ids_for_generation(project_id, prior_gen)
            self._vs.delete_ids(old_ids)
            self._lx.delete_generation(project_id, prior_gen)
            self._manifest.delete_generation(project_id, prior_gen)

        warnings.append(f"stage_cache={cache.stats()}")
        return report()

    def _embed_cached(self, chunks: list[Chunk], *, cache_root: str, batch: int = 64) -> np.ndarray:
        """Embed only chunks whose (text, profile) is not cached; commit each batch (resumable)."""
        profile_key = self._embedder.profile.key()
        dim = self._embedder.dim
        cache = EmbeddingCache(f"{cache_root}/embeddings.sqlite3")
        try:
            hashes = [text_hash(c.embed_text) for c in chunks]
            have = cache.get_many(profile_key, sorted(set(hashes)), dim)
            todo = [(h, c.embed_text) for h, c in zip(hashes, chunks, strict=True) if h not in have]
            seen: set[str] = set()
            todo = [(h, t) for h, t in todo if not (h in seen or seen.add(h))]
            for i in range(0, len(todo), batch):
                part = todo[i : i + batch]
                vecs = self._embedder.embed_passages([t for _, t in part])
                new = {h: v for (h, _), v in zip(part, vecs, strict=True)}
                cache.put_many(profile_key, new)
                have.update(new)
            return (
                np.stack([have[h] for h in hashes])
                if hashes
                else np.zeros((0, dim), dtype=np.float32)
            )
        finally:
            cache.close()

    def status(self, project_id: str):
        from .schema import StatusReport

        gen = self._manifest.active_generation(project_id)
        if gen is None:
            return StatusReport(
                project_id=project_id,
                index_generation=None,
                n_documents=0,
                n_chunks=0,
                healthy=False,
                warnings=["no active generation"],
            )
        docs = self._manifest.document_states(project_id, gen)
        states: dict[str, int] = {}
        n_chunks = 0
        for r in docs:
            states[r["state"]] = states.get(r["state"], 0) + 1
            n_chunks += r["n_chunks"] or 0
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
