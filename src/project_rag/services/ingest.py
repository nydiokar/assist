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

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

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
from ..core.ingest_policy import (
    counts_by_state,
    coverage_pct,
    embed_abort_reason,
    exact_duplicates,
    exclusion_reason,
    failed_docs,
    newly_failed,
)
from ..domain.config import ChunkingConfig, Config, ParsingConfig
from ..domain.errors import (
    ProviderContractViolation,
    ProviderTimeout,
    ProviderUnavailable,
    StoreError,
)
from ..domain.models import (
    Chunk,
    ChunkArtifact,
    DiscoveredFile,
    DocState,
    DocumentRecord,
    FileReport,
    IngestReport,
    ParseArtifact,
    ParsedDocument,
    RunReport,
    StatusReport,
)
from ..ports.cache import ArtifactStore, EmbeddingCache
from ..ports.catalog import Catalog
from ..ports.clock import Clock, IdGen
from ..ports.corpus import CorpusReader
from ..ports.embedder import Embedder
from ..ports.lexical import LexicalIndex
from ..ports.parser import DocumentParser
from ..ports.report import ReportSink
from ..ports.store import UnitOfWork
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
            # The key is content-only: bind the artifact to THIS occurrence, or a renamed file or
            # a same-bytes copy would cite the path it was first parsed under (F24).
            if cached.source_path != source_path or cached.doc.source_path != source_path:
                return cached.model_copy(
                    update={
                        "source_path": source_path,
                        "doc": cached.doc.model_copy(update={"source_path": source_path}),
                    }
                )
            return cached
        self.parse_misses += 1
        doc = registry.parse(abs_path, source_path, ext)
        art = ParseArtifact(
            artifact_id=key, source_path=source_path, source_content_hash=content_hash, doc=doc
        )
        if doc.state != DocState.FAILED:
            # A failure may be transient (OCR timeout, missing tool): never cache it, so the
            # next run retries the document (S4).
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
        # Chunks cached before the F24 fix may carry a stale source_path: treat them as a miss.
        if cached is not None and all(
            c.source_path == parse_art.doc.source_path for c in cached.chunks
        ):
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


class IngestAborted(StoreError):
    """A systemic failure (e.g. the embed server is down for most documents): nothing is
    published and the previous generation stays active."""


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
        store: UnitOfWork,
        clock: Clock,
        ids: IdGen,
        reports: ReportSink,
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
        self._store = store
        self._clock = clock
        self._ids = ids
        self._reports = reports

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

        run = _RunClock(self._clock.now().isoformat())
        with run.stage("discover"):
            listed = self._corpus.list_files(root_path)
            excluded = [(d, r) for d in listed if (r := exclusion_reason(d, self._cfg.ingest))]
            excluded_paths = {d.source_path for d, _ in excluded}
            discovered = [d for d in listed if d.source_path not in excluded_paths]
            duplicate_of = (
                exact_duplicates(discovered, tuple(self._cfg.classification.reference_prefixes))
                if self._cfg.ingest.collapse_exact_duplicates
                else {}
            )
        supported = self._parser.supported_extensions()
        pipeline = pipeline_fingerprint(
            self._cfg.parsing, self._cfg.chunking, self._cfg.classification, profile.key()
        )
        # A collapsed duplicate changes what is indexed, so it is part of the generation identity
        # (a corpus without duplicates keeps the ids it had before S5).
        new_gen = generation_id(
            project_id,
            [
                d.content_hash
                + d.source_path
                + (f"=>{duplicate_of[d.source_path]}" if d.source_path in duplicate_of else "")
                for d in discovered
            ],
            pipeline,
        )
        prior_gen = self._catalog.active_generation(project_id)
        prior_hashes = self._catalog.prior_hashes(project_id, prior_gen) if prior_gen else {}
        # Snapshot before publishing: phase 2 deletes the prior generation's rows.
        prior_states = (
            {r.source_path: r.state for r in self._catalog.document_states(project_id, prior_gen)}
            if prior_gen
            else {}
        )

        files: list[FileReport] = []
        warnings: list[str] = []
        changes = diff_corpus(prior_hashes, discovered)
        counts: dict[str, int] = {
            "added": len(changes.added),
            "modified": len(changes.modified),
            "unchanged": len(changes.unchanged),
            "deleted": len(changes.deleted),
        }

        def report(published: bool, embed_stats: tuple[int, int, int] = (0, 0, 0)) -> IngestReport:
            return IngestReport(
                project_id=project_id,
                index_generation=new_gen,
                dry_run=dry_run,
                embedding_profile_key=profile.key(),
                files=files,
                warnings=warnings,
                run=self._run_report(
                    run,
                    project_id,
                    prior_states,
                    prior_gen,
                    new_gen,
                    published,
                    dry_run,
                    n_listed=len(listed),
                    excluded=excluded,
                    files=files,
                    embed_stats=embed_stats,
                    warnings=warnings,
                ),
                **counts,
            )

        # Short-circuit true idempotency: identical corpus AND same generation already active.
        if prior_gen == new_gen and not dry_run:
            counts = {"added": 0, "modified": 0, "unchanged": len(discovered), "deleted": 0}
            warnings.append("no changes: active generation unchanged")
            prior_docs = {
                r.source_path: r for r in self._catalog.document_states(project_id, new_gen)
            }
            for d in discovered:
                prev = prior_docs.get(d.source_path)
                files.append(
                    FileReport(
                        source_path=d.source_path,
                        state=_state(prev.state) if prev else _default_state(d.ext, supported),
                        source_id=source_id(d.source_path),
                        version_id=version_id(d.content_hash, self._cfg.parsing.backend),
                        n_chunks=prev.n_chunks if prev else 0,
                        diagnostics=[prev.diagnostics] if prev and prev.diagnostics else [],
                    )
                )
            return self._finish(report(published=False))

        stages = StageRunner(self._artifacts, project_id)
        token_counter = self._embedder.token_count
        doc_chunks: dict[str, list[Chunk]] = {}
        file_index: dict[str, int] = {}
        for d in discovered:
            sid = source_id(d.source_path)
            vid = version_id(d.content_hash, self._cfg.parsing.backend)

            if d.source_path in duplicate_of:
                files.append(
                    FileReport(
                        source_path=d.source_path,
                        state=DocState.DUPLICATE,
                        source_id=sid,
                        version_id=vid,
                        diagnostics=[f"duplicate of {duplicate_of[d.source_path]}"],
                    )
                )
                continue

            if d.ext not in supported:
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

            # Per-document isolation (S4): an unexpected parser/chunker crash fails this
            # document only, with the reason, and the run continues.
            try:
                with run.stage("parse"):
                    parse_art = stages.get_or_parse(
                        abs_path=d.abs_path,
                        source_path=d.source_path,
                        ext=d.ext,
                        content_hash=d.content_hash,
                        registry=self._parser,
                        parser_cfg=self._cfg.parsing,
                    )
            except Exception as e:  # noqa: BLE001 - per-document isolation boundary
                files.append(_failed_report(d.source_path, sid, vid, "parse crashed", e))
                continue
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

            try:
                with run.stage("chunk"):
                    chunk_art = stages.get_or_chunk(
                        parse_art=parse_art,
                        project_id=project_id,
                        source_id=sid,
                        version_id=vid,
                        source_type=d.ext.lstrip("."),
                        chunk_cfg=self._cfg.chunking,
                        token_counter=token_counter,
                    )
            except Exception as e:  # noqa: BLE001 - per-document isolation boundary
                files.append(_failed_report(d.source_path, sid, vid, "chunking crashed", e))
                continue
            doc_class = classify_doc(
                d.source_path, tuple(self._cfg.classification.reference_prefixes)
            )
            chunks = [c.model_copy(update={"doc_class": doc_class}) for c in chunk_art.chunks]
            doc_chunks[d.source_path] = chunks
            diags = list(doc.diagnostics)
            if any(c.truncated for c in chunks):  # A01 §F: silent truncation is a failed check
                diags.append(
                    f"TRUNCATION: {sum(c.truncated for c in chunks)} chunk(s) exceeded "
                    f"max_tokens={self._cfg.chunking.max_tokens}"
                )
            file_index[d.source_path] = len(files)
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

        fs_chunks: list[Chunk] = []
        if self._cfg.classification.filesystem_evidence:
            # Filesystem-presence evidence (DECISIONS 2026-10-28): synthesized from the walk.
            fs_chunks = build_filesystem_chunks(project_id, discovered)
            if fs_chunks:
                warnings.append(f"filesystem_evidence: {len(fs_chunks)} presence chunk(s)")

        if dry_run:
            warnings.append("dry run: no writes performed")
            return self._finish(report(published=False))

        # Embed per document (isolation): a document whose vectors fail is FAILED with the
        # reason; a systemic failure aborts the run before anything is published.
        with run.stage("embed"):
            vectors, embed_failures, embed_stats = self._embed_documents(
                [*doc_chunks.items(), *(("", [c]) for c in fs_chunks)]
            )
        for path, reason in embed_failures.items():
            if path in file_index:
                i = file_index[path]
                files[i] = files[i].model_copy(
                    update={
                        "state": DocState.FAILED,
                        "n_chunks": 0,
                        "diagnostics": [*files[i].diagnostics, reason],
                    }
                )
                doc_chunks.pop(path, None)
        abort = embed_abort_reason(len(embed_failures), embed_stats[2], self._cfg.ingest)
        if abort:
            warnings.append("ABORTED: " + abort)
            self._finish(report(published=False, embed_stats=embed_stats))
            raise IngestAborted(abort)
        if embed_failures:
            warnings.append(f"embedding failed for {len(embed_failures)} document(s)")

        staged_chunks = [c for cs in doc_chunks.values() for c in cs]
        staged_chunks += [c for c in fs_chunks if text_hash(c.embed_text) in vectors]
        embeddings = (
            np.stack([vectors[text_hash(c.embed_text)] for c in staged_chunks])
            if staged_chunks
            else np.zeros((0, self._embedder.dim), dtype=np.float32)
        )
        content_hashes = {d.source_path: d.content_hash for d in discovered}
        records = [
            DocumentRecord(
                project_id=project_id,
                source_id=fr.source_id,
                source_path=fr.source_path,
                content_hash=content_hashes.get(fr.source_path, ""),
                version_id=fr.version_id or "",
                state=fr.state,
                generation=new_gen,
                n_chunks=fr.n_chunks,
                diagnostics="; ".join(fr.diagnostics),
            )
            for fr in files
        ]
        with run.stage("publish"):
            self._publish(project_id, new_gen, prior_gen, staged_chunks, embeddings, records)

        warnings.append(f"stage_cache={stages.stats()}")
        return self._finish(report(published=True, embed_stats=embed_stats))

    def _embed_documents(
        self, docs: list[tuple[str, list[Chunk]]], *, batch: int = 64
    ) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, str], tuple[int, int, int]]:
        """Vectors by text hash for every chunk whose document embedded, failures by document,
        and (computed, cached, documents that needed new vectors) counts. Uncached texts are embedded per document, in batches, and
        committed to the cache as they finish (resumable)."""
        profile_key = self._embedder.profile.key()
        dim = self._embedder.dim
        cache = self._open_embedding_cache()
        failures: dict[str, str] = {}
        computed = 0
        needing = 0
        try:
            all_hashes = sorted({text_hash(c.embed_text) for _, cs in docs for c in cs})
            have = cache.get_many(profile_key, all_hashes, dim)
            cached = len(have)
            for path, chunks in docs:
                todo: list[tuple[str, str]] = []
                seen: set[str] = set()
                for c in chunks:
                    h = text_hash(c.embed_text)
                    if h in have or h in seen:
                        continue
                    seen.add(h)
                    todo.append((h, c.embed_text))
                needing += bool(todo)
                try:
                    for i in range(0, len(todo), batch):
                        part = todo[i : i + batch]
                        vecs = self._embedder.embed_passages([t for _, t in part])
                        new = {h: v for (h, _), v in zip(part, vecs, strict=True)}
                        cache.put_many(profile_key, new)
                        have.update(new)
                        computed += len(new)
                except (
                    ProviderUnavailable,
                    ProviderTimeout,
                    ProviderContractViolation,
                    ValueError,
                ) as e:
                    failures[path or "(filesystem evidence)"] = (
                        f"embed failed: {type(e).__name__}: {e}"
                    )
            return have, failures, (computed, cached, needing)
        finally:
            cache.close()

    def _run_report(
        self,
        run: _RunClock,
        project_id: str,
        prior_states: dict[str, str],
        prior_gen: str | None,
        new_gen: str,
        published: bool,
        dry_run: bool,
        *,
        n_listed: int,
        excluded: list[tuple[DiscoveredFile, str]],
        files: list[FileReport],
        embed_stats: tuple[int, int, int],
        warnings: list[str],
    ) -> RunReport:
        current_failed = failed_docs(files)
        ready = [f for f in files if f.state == DocState.READY]
        return RunReport(
            run_id=self._ids.new_id(),
            project_id=project_id,
            started_at=run.started_at,
            finished_at=self._clock.now().isoformat(),
            prior_generation=prior_gen,
            new_generation=new_gen,
            published=published,
            dry_run=dry_run,
            n_discovered=n_listed,
            n_excluded=len(excluded),
            n_in_scope=len(files),
            counts_by_state=counts_by_state(files),
            coverage_pct=coverage_pct(files),
            failed=current_failed,
            newly_failed=newly_failed(current_failed, prior_states),
            ocr_indexed=[f.source_path for f in ready if any("ocr_" in x for x in f.diagnostics)],
            antiword_indexed=[f.source_path for f in ready if "parser=antiword" in f.diagnostics],
            duplicate_of={
                f.source_path: f.diagnostics[0].removeprefix("duplicate of ")
                for f in files
                if f.state == DocState.DUPLICATE and f.diagnostics
            },
            embeddings_computed=embed_stats[0],
            embeddings_cached=embed_stats[1],
            stage_seconds=run.seconds(),
            warnings=list(warnings),
        )

    def _finish(self, report: IngestReport) -> IngestReport:
        if report.run is not None:
            self._catalog.record_run(report.run)
            self._reports.write(report.run)
        return report

    def _publish(
        self,
        project_id: str,
        new_gen: str,
        prior_gen: str | None,
        chunks: list[Chunk],
        embeddings: npt.NDArray[np.float32],
        records: list[DocumentRecord],
    ) -> None:
        """Two-phase publish (S2, D-3 = Chroma). The active-pointer flip is the commit point.

        0. Journal the staging (committed): from here on a crash leaves a trace the GC can use.
        1. One store transaction: purge leftovers of `new_gen`, stage vectors + lexical + catalog
           rows, verify the counts, set the profile, flip the pointer. Indexes inside the store
           (FTS5, catalog, sqlite-vec) roll back together; an index outside it (Chroma, memory)
           may keep staged vectors, which the journal lets the GC remove.
        2. Delete the superseded generation. A crash here leaves the new generation active and
           complete; `stale_generations()` finds the old rows at the next start.
        """
        profile = self._embedder.profile
        collection = self._cfg.vector_store.collection
        self._catalog.begin_staging(project_id, new_gen, self._clock.now())
        with self._store.transaction():
            self._drop_generation(project_id, new_gen)
            if chunks:
                self._vector.add(new_gen, chunks, embeddings)
                self._lexical.add_chunks(new_gen, chunks)
                self._catalog.record_chunks(
                    project_id, new_gen, [(c.chunk_id, c.source_id, c.version_id) for c in chunks]
                )
            for record in records:
                self._catalog.upsert_document(record)
            self._verify_staged(project_id, new_gen, len(chunks))
            self._catalog.set_project_profile(project_id, profile, collection)
            self._catalog.publish_generation(project_id, new_gen)
        if prior_gen and prior_gen != new_gen:
            with self._store.transaction():
                self._drop_generation(project_id, prior_gen)

    def _verify_staged(self, project_id: str, generation: str, expected: int) -> None:
        counts = {
            "vector": self._vector.count(project_id, generation),
            "lexical": self._lexical.count(project_id, generation),
            "catalog": len(self._catalog.chunk_ids_for_generation(project_id, generation)),
        }
        wrong = {k: v for k, v in counts.items() if v != expected}
        if wrong:
            raise StoreError(f"staged {generation}: expected {expected} chunks, got {wrong}")

    def _drop_generation(self, project_id: str, generation: str) -> None:
        self._vector.delete_generation(project_id, generation)
        self._lexical.delete_generation(project_id, generation)
        self._catalog.delete_generation(project_id, generation)

    def collect_garbage(self) -> list[tuple[str, str]]:
        """Startup GC (S2): remove every non-active generation from all indexes.

        Covers crashed stagings (journal rows, whose Chroma/memory vectors may exist) and
        superseded generations whose phase-2 delete did not finish. Returns what was removed.
        Single-writer assumption: no ingest may run while this executes.
        """
        stale = self._catalog.stale_generations()
        for project_id, generation in stale:
            with self._store.transaction():
                self._drop_generation(project_id, generation)
        return stale

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


class _RunClock:
    """Accumulates wall-clock seconds per ingest stage for the run report."""

    def __init__(self, started_at: str) -> None:
        self.started_at = started_at
        self._seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._seconds[name] = self._seconds.get(name, 0.0) + time.perf_counter() - t0

    def seconds(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self._seconds.items()}


def _state(value: str) -> DocState:
    try:
        return DocState(value)
    except ValueError:
        return DocState.FAILED


def _default_state(ext: str, supported: frozenset[str]) -> DocState:
    return DocState.READY if ext in supported else DocState.UNSUPPORTED


def _failed_report(path: str, sid: str, vid: str, what: str, e: BaseException) -> FileReport:
    return FileReport(
        source_path=path,
        state=DocState.FAILED,
        source_id=sid,
        version_id=vid,
        diagnostics=[f"{what}: {type(e).__name__}: {e}"],
    )
