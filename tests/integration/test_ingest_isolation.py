"""S4 exit checks on fixtures: per-document isolation (corrupt PDF, OCR crash, embed timeout),
exclusions, the systemic-failure abort, and the persisted run report."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from project_rag.adapters.clock_system import SequentialIdGen, SystemClock
from project_rag.adapters.corpus_local_fs import LocalFsCorpus
from project_rag.adapters.embed_hash import HashEmbedder
from project_rag.adapters.parse_registry import ParserRegistry
from project_rag.adapters.report_json import JsonReportWriter
from project_rag.adapters.sqlite.catalog import SqliteCatalog
from project_rag.adapters.sqlite.db import SqliteDb
from project_rag.adapters.sqlite.embedding_cache import SqliteEmbeddingCache
from project_rag.adapters.sqlite.fts5 import Fts5LexicalIndex
from project_rag.adapters.stage_cache_fs import FsArtifactStore
from project_rag.adapters.vector_memory import InMemoryVectorStore
from project_rag.domain.config import Config, with_overrides
from project_rag.domain.errors import OcrFailed, ProviderTimeout
from project_rag.domain.models import DocState, EmbeddingProfile, OcrText
from project_rag.services.ingest import IngestAborted, IngestService
from project_rag.services.parsing import OcrAwareParser
from tests.conftest import FIXTURE_CORPUS

PID = "demo"
POISON = "POISONED-FOR-EMBED"


class FlakyEmbedder:
    """Hash embedder that times out on any batch containing a marker text."""

    def __init__(self, inner: HashEmbedder, poison: str, *, always: bool = False) -> None:
        self._inner = inner
        self._poison = poison
        self._always = always

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def profile(self) -> EmbeddingProfile:
        return self._inner.profile

    @property
    def max_input_tokens(self) -> int:
        return self._inner.max_input_tokens

    def token_count(self, text: str) -> int:
        return self._inner.token_count(text)

    def embed_passages(self, texts: list[str]) -> npt.NDArray[np.float32]:
        if self._always or any(self._poison in t for t in texts):
            raise ProviderTimeout("embed server timed out (injected)")
        return self._inner.embed_passages(texts)

    def embed_queries(self, texts: list[str]) -> npt.NDArray[np.float32]:
        return self._inner.embed_queries(texts)


class FakeOcr:
    """Reads text from `<image>.txt` beside the image; images named `crash*` raise."""

    def ocr_image(self, path: str) -> OcrText:
        p = Path(path)
        if p.name.startswith("crash"):
            raise OcrFailed("tesseract exit 1: segfault (injected)")
        side = p.with_suffix(p.suffix + ".ocr")
        text = side.read_text(encoding="utf-8") if side.is_file() else ""
        return OcrText(text=text, mean_conf=90.0 if text else 0.0)

    def ocr_pdf(self, path: str) -> list[str]:
        return ["Signed attendance sheet for the March training in the village of Breza."]


def _corpus(tmp: Path) -> Path:
    root = tmp / "corpus"
    shutil.copytree(FIXTURE_CORPUS, root)
    (root / "broken.pdf").write_bytes(b"%PDF-1.4\n garbage that is not a pdf \x00\xff" * 20)
    # Distinct bytes per image: the parse cache is keyed by content hash (identical bytes parse
    # identically), and this fake OCR reads a per-file sidecar.
    (root / "crash_scan.png").write_bytes(b"\x89PNG fake crash")
    poster = root / "poster.png"
    poster.write_bytes(b"\x89PNG fake poster")
    (root / "poster.png.ocr").write_text(
        "Национална среща на младежите 2025. Регистрация до 15 май на сайта на проекта.",
        encoding="utf-8",
    )
    (root / "event_photo.jpg").write_bytes(b"\xff\xd8 fake")  # no OCR text -> NO_TEXT
    (root / "poisoned.md").write_text(f"# Poison\n\n{POISON} paragraph.\n", encoding="utf-8")
    (root / "webinar.mp4").write_bytes(b"\x00" * 64)  # excluded media
    (root / "~$draft.docx").write_bytes(b"lock")  # Office lock file: ignored by discovery
    (root / "notes.tmp").write_bytes(b"tmp")
    return root


def _service(tmp: Path, *, always_fail: bool = False) -> tuple[IngestService, SqliteDb, Path]:
    cfg = with_overrides(
        Config(),
        {
            "parsing.ocr.enabled": True,
            "parsing.ocr.min_chars": 20,
            "vector_store.persist_path": "",
        },
    )
    db = SqliteDb(tmp / "state" / "store.sqlite3")
    runs = tmp / "state" / "runs"
    svc = IngestService(
        cfg,
        corpus=LocalFsCorpus(),
        parser=OcrAwareParser(ParserRegistry(cfg.parsing), FakeOcr(), cfg.parsing),
        embedder=FlakyEmbedder(HashEmbedder(cfg.embedding), POISON, always=always_fail),
        vector=InMemoryVectorStore(cfg.vector_store),
        lexical=Fts5LexicalIndex(db),
        catalog=SqliteCatalog(db),
        artifacts=FsArtifactStore(tmp / "state" / "cache"),
        open_embedding_cache=lambda: SqliteEmbeddingCache(tmp / "state" / "cache" / "emb.sqlite3"),
        store=db,
        clock=SystemClock(),
        ids=SequentialIdGen("run"),
        reports=JsonReportWriter(runs),
    )
    return svc, db, runs


def test_faults_are_isolated_and_the_run_publishes(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    svc, db, runs = _service(tmp_path)
    report = svc.ingest(PID, str(root))
    by = {f.source_path: f for f in report.files}

    assert by["broken.pdf"].state == DocState.FAILED  # corrupt PDF
    assert by["crash_scan.png"].state == DocState.FAILED  # OCR crash
    assert "ocr failed" in by["crash_scan.png"].diagnostics[0]
    assert by["poisoned.md"].state == DocState.FAILED  # embed timeout
    assert any("ProviderTimeout" in d for d in by["poisoned.md"].diagnostics)
    assert by["event_photo.jpg"].state == DocState.NO_TEXT  # photo without text
    assert by["poster.png"].state == DocState.READY  # OCR'd image indexed
    for ok in ("proposal.md", "memo_bg.md", "indicators.xlsx", "registrations.csv"):
        assert by[ok].state == DocState.READY, ok

    gen = report.index_generation
    catalog = SqliteCatalog(db)
    assert catalog.active_generation(PID) == gen  # published despite the faults
    states = {r.source_path: r.state for r in catalog.document_states(PID, gen)}
    assert states["poisoned.md"] == "FAILED" and states["poster.png"] == "READY"
    assert "webinar.mp4" not in states and "notes.tmp" not in states  # excluded

    run = report.run
    assert run is not None and run.published
    # The media file is excluded by config; lock/temp files never reach the catalog either
    # (discovery drops them).
    assert run.n_excluded == 1 and run.counts_by_state["FAILED"] == 3
    assert run.counts_by_state["NO_TEXT"] == 1
    assert {d.source_path for d in run.newly_failed} == {
        "broken.pdf",
        "crash_scan.png",
        "poisoned.md",
        "event_photo.jpg",
    }
    assert run.ocr_indexed == ["poster.png", "scanned_placeholder.pdf"]
    assert by["scanned_placeholder.pdf"].state == DocState.READY  # scanned PDF via OCR
    assert 0 < run.coverage_pct < 100
    assert {"discover", "parse", "chunk", "embed", "publish"} <= set(run.stage_seconds)
    stored = catalog.runs(PID)
    assert stored and stored[0].run_id == run.run_id
    files = list(runs.glob(f"*_{PID}_{run.run_id}.json"))
    assert len(files) == 1 and json.loads(files[0].read_text(encoding="utf-8"))["published"]
    db.close()


def test_second_run_reports_no_new_failures(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    svc, db, _ = _service(tmp_path)
    svc.ingest(PID, str(root))
    (root / "extra.md").write_text("# Extra\n\nA new note about gardens.\n", encoding="utf-8")
    second = svc.ingest(PID, str(root))
    assert second.run is not None and second.run.published
    assert second.run.newly_failed == []  # the same four still fail; none is new
    assert len(second.run.failed) == 4
    db.close()


def test_systemic_embed_failure_aborts_and_keeps_previous_generation(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    svc, db, _ = _service(tmp_path)
    g1 = svc.ingest(PID, str(root)).index_generation
    for name in ("proposal.md", "memo_bg.md", "progress_report.md", "registrations.csv"):
        with (root / name).open("a", encoding="utf-8") as f:
            f.write("\n\nAmended after the outage started.\n")
    down, _, _ = _service(tmp_path, always_fail=True)
    with pytest.raises(IngestAborted, match="nothing published"):
        down.ingest(PID, str(root))
    assert SqliteCatalog(db).active_generation(PID) == g1
    runs = SqliteCatalog(db).runs(PID)
    assert runs[0].published is False and any("ABORTED" in w for w in runs[0].warnings)
    db.close()


def test_parser_crash_is_isolated(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    svc, db, _ = _service(tmp_path)
    real = svc._parser.parse  # type: ignore[attr-defined]

    def crashing(abs_path: str, source_path: str, ext: str):
        if source_path == "memo_bg.md":
            raise RuntimeError("parser bug (injected)")
        return real(abs_path, source_path, ext)

    svc._parser.parse = crashing  # type: ignore[attr-defined]
    report = svc.ingest(PID, str(root))
    by = {f.source_path: f for f in report.files}
    assert by["memo_bg.md"].state == DocState.FAILED
    assert "parse crashed: RuntimeError" in by["memo_bg.md"].diagnostics[0]
    assert by["proposal.md"].state == DocState.READY
    assert report.run is not None and report.run.published
    db.close()
