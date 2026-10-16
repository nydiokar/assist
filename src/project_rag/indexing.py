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

from .chunking import CHUNKER_VERSION, chunk_document
from .config import Config
from .discovery import discover
from .embeddings import EmbedderProtocol
from .lexical_store import LexicalStore
from .manifest import Manifest
from .parsers import ParserRegistry
from .schema import Chunk, DocState, FileReport, IngestReport, ParsedDocument
from .vector_store import VectorStoreProtocol


def _source_id(source_path: str) -> str:
    return "src_" + hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:20]


def _version_id(content_hash: str, parser_backend: str) -> str:
    raw = f"{content_hash}|{parser_backend}|{CHUNKER_VERSION}"
    return "v_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _generation_id(project_id: str, hashes: list[str]) -> str:
    raw = project_id + "|" + "|".join(sorted(hashes))
    return "g_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class IndexingService:
    def __init__(self, cfg: Config, embedder: EmbedderProtocol,
                 vector_store: VectorStoreProtocol, lexical_store: LexicalStore,
                 manifest: Manifest, registry: ParserRegistry) -> None:
        self._cfg = cfg
        self._embedder = embedder
        self._vs = vector_store
        self._lx = lexical_store
        self._manifest = manifest
        self._registry = registry

    def ingest(self, project_id: str, root_path: str, *, dry_run: bool = False) -> IngestReport:
        profile = self._embedder.profile
        collection = self._cfg.vector_store.collection

        problems = self._manifest.check_compatibility(project_id, profile, collection)
        if problems:
            return IngestReport(project_id=project_id, index_generation="",
                                embedding_profile_key=profile.key(),
                                warnings=["REFUSED: " + p for p in problems])

        if profile.dim != self._embedder.dim:  # defensive
            return IngestReport(project_id=project_id, index_generation="",
                                warnings=["REFUSED: embedder dim != profile dim"])

        discovered = discover(root_path)
        new_gen = _generation_id(project_id, [d.content_hash + d.source_path for d in discovered])
        prior_gen = self._manifest.active_generation(project_id)
        prior_hashes = (self._manifest.prior_hashes(project_id, prior_gen)
                        if prior_gen else {})

        report = IngestReport(project_id=project_id, index_generation=new_gen,
                              dry_run=dry_run, embedding_profile_key=profile.key())

        # Short-circuit true idempotency: identical corpus AND same generation already active.
        if prior_gen == new_gen and not dry_run:
            report.unchanged = len(discovered)
            report.warnings.append("no changes: active generation unchanged")
            for d in discovered:
                report.files.append(FileReport(source_path=d.source_path,
                                                state=DocState.READY if d.supported
                                                else DocState.UNSUPPORTED,
                                                source_id=_source_id(d.source_path),
                                                version_id=_version_id(d.content_hash,
                                                                       self._cfg.parsing.backend)))
            return report

        staged_chunks: list[Chunk] = []
        for d in discovered:
            sid = _source_id(d.source_path)
            vid = _version_id(d.content_hash, self._cfg.parsing.backend)
            prior = prior_hashes.get(d.source_path)
            changed = prior is None or prior[0] != d.content_hash
            if prior is None:
                report.added += 1
            elif changed:
                report.modified += 1
            else:
                report.unchanged += 1

            if not d.supported:
                report.files.append(FileReport(source_path=d.source_path,
                                                state=DocState.UNSUPPORTED, source_id=sid,
                                                version_id=vid,
                                                diagnostics=[f"unsupported ext {d.ext}"]))
                continue

            doc: ParsedDocument = self._registry.parse(d.abs_path, d.source_path, d.ext)
            if doc.state not in {DocState.PARSED}:
                report.files.append(FileReport(source_path=d.source_path, state=doc.state,
                                                source_id=sid, version_id=vid,
                                                diagnostics=doc.diagnostics))
                continue

            chunks = chunk_document(doc, project_id=project_id, source_id=sid, version_id=vid,
                                    source_type=d.ext.lstrip("."), cfg=self._cfg.chunking)
            staged_chunks.extend(chunks)
            report.files.append(FileReport(source_path=d.source_path, state=DocState.READY,
                                            source_id=sid, version_id=vid,
                                            n_blocks=len(doc.blocks), n_chunks=len(chunks),
                                            diagnostics=doc.diagnostics))

        deleted = set(prior_hashes) - {d.source_path for d in discovered}
        report.deleted = len(deleted)

        if dry_run:
            report.warnings.append("dry run: no writes performed")
            return report

        # Stage: embed + write new generation.
        if staged_chunks:
            embeddings = self._embedder.embed_passages([c.embed_text for c in staged_chunks])
            self._vs.add(new_gen, staged_chunks, embeddings)
            self._lx.add_chunks(new_gen, staged_chunks)
            self._manifest.record_chunks(
                project_id, new_gen,
                [(c.chunk_id, c.source_id, c.version_id) for c in staged_chunks])

        for fr in report.files:
            content_hash = next((d.content_hash for d in discovered
                                 if d.source_path == fr.source_path), "")
            self._manifest.upsert_document(
                project_id=project_id, source_id=fr.source_id, source_path=fr.source_path,
                content_hash=content_hash, version_id=fr.version_id or "", state=fr.state,
                generation=new_gen, n_chunks=fr.n_chunks,
                diagnostics="; ".join(fr.diagnostics))

        # Publish new active generation, then remove superseded derivatives.
        self._manifest.set_project_profile(project_id, profile, collection)
        self._manifest.publish_generation(project_id, new_gen)
        if prior_gen and prior_gen != new_gen:
            old_ids = self._manifest.chunk_ids_for_generation(project_id, prior_gen)
            self._vs.delete_ids(old_ids)
            self._lx.delete_generation(project_id, prior_gen)
            self._manifest.delete_generation(project_id, prior_gen)

        return report

    def status(self, project_id: str):
        from .schema import StatusReport  # noqa: PLC0415

        gen = self._manifest.active_generation(project_id)
        if gen is None:
            return StatusReport(project_id=project_id, index_generation=None, n_documents=0,
                                n_chunks=0, healthy=False, warnings=["no active generation"])
        docs = self._manifest.document_states(project_id, gen)
        states: dict[str, int] = {}
        n_chunks = 0
        for r in docs:
            states[r["state"]] = states.get(r["state"], 0) + 1
            n_chunks += r["n_chunks"] or 0
        return StatusReport(project_id=project_id, index_generation=gen, n_documents=len(docs),
                            n_chunks=n_chunks, states=states,
                            embedding_profile_key=self._embedder.profile.key(),
                            collection=self._cfg.vector_store.collection, healthy=True)
