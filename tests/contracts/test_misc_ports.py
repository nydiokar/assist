"""Contracts for the small ports: stage caches, feedback, telemetry, clock/ids, OCR."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from project_rag.adapters.clock_system import FixedClock, SequentialIdGen, SystemClock, UuidIdGen
from project_rag.adapters.feedback_memory import MemoryFeedbackSink
from project_rag.adapters.ocr_null import NullOcr
from project_rag.adapters.sqlite.embedding_cache import SqliteEmbeddingCache
from project_rag.adapters.stage_cache_fs import FsArtifactStore
from project_rag.adapters.telemetry_null import NullTelemetry
from project_rag.domain.errors import OcrFailed
from project_rag.domain.models import (
    ChunkArtifact,
    DocState,
    FeedbackSignal,
    ParseArtifact,
    ParsedDocument,
)
from project_rag.ports.cache import ArtifactStore, EmbeddingCache
from project_rag.ports.clock import Clock, IdGen
from project_rag.ports.feedback import FeedbackSink
from project_rag.ports.ocr import OcrEngine
from project_rag.ports.telemetry import Telemetry
from tests.contracts.conftest import make_chunk

# ---- ArtifactStore ----


@pytest.fixture(params=["fs"])
def artifacts(request: pytest.FixtureRequest, tmp_path: Path) -> ArtifactStore:
    return FsArtifactStore(tmp_path / "cache")


def test_artifact_store_roundtrip_and_miss(artifacts: ArtifactStore) -> None:
    assert artifacts.get_parse("k") is None
    pa = ParseArtifact(
        artifact_id="k",
        source_path="a.md",
        source_content_hash="h",
        doc=ParsedDocument(source_path="a.md", state=DocState.PARSED),
    )
    artifacts.put_parse(pa)
    assert artifacts.get_parse("k") == pa
    ca = ChunkArtifact(artifact_id="ck", parsed_artifact_id="k", chunks=[make_chunk("c1", "t")])
    assert artifacts.get_chunk("p1", "ck") is None
    artifacts.put_chunk("p1", ca)
    assert artifacts.get_chunk("p1", "ck") == ca
    assert artifacts.get_chunk("p2", "ck") is None  # chunk artifacts are per project


# ---- EmbeddingCache ----


@pytest.fixture(params=["sqlite"])
def emb_cache(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[EmbeddingCache]:
    c = SqliteEmbeddingCache(tmp_path / "emb.sqlite3")
    yield c
    c.close()


def test_embedding_cache_roundtrip_scoped_by_profile_and_dim(emb_cache: EmbeddingCache) -> None:
    v = np.arange(4, dtype=np.float32)
    emb_cache.put_many("prof", {"h1": v})
    got = emb_cache.get_many("prof", ["h1", "h2"], 4)
    assert list(got) == ["h1"] and np.array_equal(got["h1"], v)
    assert emb_cache.get_many("other", ["h1"], 4) == {}
    assert emb_cache.get_many("prof", ["h1"], 8) == {}


# ---- FeedbackSink ----

T0 = datetime(2026, 11, 11, tzinfo=UTC)


@pytest.fixture(params=["memory"])
def sink(request: pytest.FixtureRequest) -> FeedbackSink:
    return MemoryFeedbackSink()


def test_feedback_query_window_oldest_first(sink: FeedbackSink) -> None:
    late = FeedbackSignal(kind="empty", project_id="p1", at=T0 + timedelta(hours=2))
    early = FeedbackSignal(kind="abstain", project_id="p1", at=T0 + timedelta(hours=1))
    old = FeedbackSignal(kind="abstain", project_id="p1", at=T0 - timedelta(days=1))
    for s in (late, early, old):
        sink.record(s)
    assert sink.query(T0) == [early, late]


# ---- Telemetry ----


@pytest.fixture(params=["null"])
def telemetry(request: pytest.FixtureRequest) -> Telemetry:
    return NullTelemetry()


def test_telemetry_never_raises(telemetry: Telemetry) -> None:
    telemetry.observe("retrieve", 0.12)
    telemetry.observe("", -1.0)
    telemetry.count("rag_queries_total", {"project": "p1"})
    telemetry.count("x")


# ---- Clock / IdGen ----


@pytest.mark.parametrize("clock", [SystemClock(), FixedClock(T0)], ids=["system", "fixed"])
def test_clock_is_timezone_aware(clock: Clock) -> None:
    assert clock.now().tzinfo is not None


@pytest.mark.parametrize("ids", [UuidIdGen(), SequentialIdGen()], ids=["uuid", "sequential"])
def test_ids_are_unique(ids: IdGen) -> None:
    made = {ids.new_id() for _ in range(100)}
    assert len(made) == 100


# ---- OcrEngine ----


@pytest.mark.parametrize("ocr", [NullOcr()], ids=["null"])
def test_ocr_failure_is_a_port_error(ocr: OcrEngine) -> None:
    with pytest.raises(OcrFailed):
        ocr.ocr_image("x.png")
    with pytest.raises(OcrFailed):
        ocr.ocr_pdf("x.pdf")
