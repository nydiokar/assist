"""Crash injection for the two-phase generation publish (S2 exit check).

For every step of publish we make that step fail (raise, standing in for a kill), then check:
- before the commit point (pointer flip): the previous generation is still active and fully
  searchable, and nothing else became visible;
- after the commit point: the new generation is active and complete;
- after a restart (a fresh `wiring.build`, which runs the startup GC): no rows or vectors of any
  non-active generation are left, and the next ingest publishes normally.
One more test kills a real child process in the middle of staging.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from project_rag.domain.config import Config, with_overrides
from project_rag.domain.errors import StoreError
from project_rag.wiring import Services, build
from tests.conftest import FIXTURE_CORPUS

PID = "demo"
QUERY = "grant number Green Villages"


class Boom(RuntimeError):
    pass


def _config(tmp: Path, vector: str) -> Config:
    overrides: dict[str, object] = {
        "lexical_store.path": str(tmp / "state" / "store.sqlite3"),
        "vector_store.provider": vector,
        "vector_store.persist_path": str(tmp / "state" / "vectors"),
        "vector_store.collection": "crash_test",
    }
    return with_overrides(Config(), overrides)


def _vector_kinds() -> list[object]:
    kinds: list[object] = ["memory", "sqlite_vec"]
    try:
        import chromadb  # noqa: F401

        kinds.append("chroma")
    except ImportError:
        kinds.append(pytest.param("chroma", marks=pytest.mark.skip(reason="chromadb absent")))
    return kinds


def _corpus(tmp: Path) -> Path:
    root = tmp / "corpus"
    shutil.copytree(FIXTURE_CORPUS, root)
    return root


def _modify(root: Path) -> None:
    with (root / "proposal.md").open("a", encoding="utf-8") as f:
        f.write("\n\n## Amendment\n\nThe steering committee approved a fourth community garden.\n")


def _counts(s: Services, gen: str) -> tuple[int, int, int]:
    return (
        s.vector.count(PID, gen),
        s.lexical.count(PID, gen),
        len(s.catalog.chunk_ids_for_generation(PID, gen)),
    )


def _assert_serves(s: Services, gen: str) -> None:
    res = s.search.search_project(PID, QUERY, top_k=5)
    assert res.index_generation == gen
    assert res.results, "active generation must stay searchable"
    members = set(s.catalog.chunk_ids_for_generation(PID, gen))
    assert {r.chunk_id for r in res.results} <= members


def _raise_when(fn: Callable[..., object], gen: str | None = None) -> Callable[..., object]:
    def wrapper(*args: object, **kwargs: object) -> object:
        if gen is None or gen in args or gen in kwargs.values():
            raise Boom(fn.__name__)
        return fn(*args, **kwargs)

    return wrapper


# (label, component, method, only-when-called-with-prior-generation, after_commit_point)
STEPS: list[tuple[str, str, str, bool, bool]] = [
    ("journal", "catalog", "begin_staging", False, False),
    ("stage_vectors", "vector", "add", False, False),
    ("stage_lexical", "lexical", "add_chunks", False, False),
    ("stage_chunk_rows", "catalog", "record_chunks", False, False),
    ("stage_documents", "catalog", "upsert_document", False, False),
    ("verify_counts", "lexical", "count", False, False),
    ("pin_profile", "catalog", "set_project_profile", False, False),
    ("flip_pointer", "catalog", "publish_generation", False, False),
    ("drop_old_vectors", "vector", "delete_generation", True, True),
    ("drop_old_lexical", "lexical", "delete_generation", True, True),
    ("drop_old_catalog", "catalog", "delete_generation", True, True),
]


@pytest.mark.parametrize("vector", _vector_kinds())
@pytest.mark.parametrize(
    ("label", "component", "method", "prior_only", "after_commit"),
    STEPS,
    ids=[s[0] for s in STEPS],
)
def test_failure_at_each_publish_step(
    tmp_path: Path,
    vector: str,
    label: str,
    component: str,
    method: str,
    prior_only: bool,
    after_commit: bool,
) -> None:
    cfg = _config(tmp_path, vector)
    root = _corpus(tmp_path)
    s = build(cfg)
    s.search.register_project_root(PID, str(root))
    g1 = s.ingest.ingest(PID, str(root)).index_generation
    n1 = _counts(s, g1)
    assert n1[0] > 0 and len(set(n1)) == 1

    _modify(root)
    target = getattr(s, component)
    original = getattr(target, method)
    if label == "verify_counts":
        # The verifier reads counts; a wrong count must abort the publish.
        def wrong_count(project_id: str, generation: str) -> int:
            return original(project_id, generation) + (1 if generation != g1 else 0)

        setattr(target, method, wrong_count)
        expected_error: type[BaseException] = StoreError
    else:
        setattr(target, method, _raise_when(original, g1 if prior_only else None))
        expected_error = Boom
    with pytest.raises(expected_error):
        s.ingest.ingest(PID, str(root))
    setattr(target, method, original)

    active = s.catalog.active_generation(PID)
    if after_commit:
        assert active is not None and active != g1
        new = _counts(s, active)
        assert new[0] > 0 and len(set(new)) == 1
        _assert_serves(s, active)
    else:
        assert active == g1
        assert _counts(s, g1) == n1
        _assert_serves(s, g1)
    s.close()

    # Restart: the startup GC removes every non-active generation.
    s2 = build(cfg)
    s2.search.register_project_root(PID, str(root))
    assert s2.catalog.stale_generations() == []
    keep = s2.catalog.active_generation(PID)
    assert keep is not None
    for gen in {g1, keep}:
        expect = _counts(s2, keep) if gen == keep else (0, 0, 0)
        assert _counts(s2, gen) == expect
    if not after_commit:
        assert s2.gc_removed or vector == "sqlite_vec" or label == "journal"
    _assert_serves(s2, keep)
    # And the next ingest completes normally.
    report = s2.ingest.ingest(PID, str(root))
    final = s2.catalog.active_generation(PID)
    assert final == report.index_generation and final != g1
    assert s2.catalog.stale_generations() == []
    assert _counts(s2, g1) == (0, 0, 0)
    s2.close()


def test_sqlite_vec_staging_rolls_back_inside_one_transaction(tmp_path: Path) -> None:
    """D-3 rule input: with every index in the one store, a failed publish leaves no staged rows
    at all (only the journal entry), even before any GC."""
    cfg = _config(tmp_path, "sqlite_vec")
    root = _corpus(tmp_path)
    s = build(cfg)
    g1 = s.ingest.ingest(PID, str(root)).index_generation
    _modify(root)
    original = s.catalog.publish_generation
    s.catalog.publish_generation = _raise_when(original)  # type: ignore[method-assign]
    with pytest.raises(Boom):
        s.ingest.ingest(PID, str(root))
    stale = s.catalog.stale_generations()
    assert len(stale) == 1 and stale[0][1] != g1  # the journal row only
    assert _counts(s, stale[0][1]) == (0, 0, 0)
    assert s.catalog.active_generation(PID) == g1
    s.close()


CHILD = textwrap.dedent(
    """
    import os, sys
    sys.path.insert(0, {src!r})
    from project_rag.domain.config import Config, with_overrides
    from project_rag.wiring import build
    cfg = with_overrides(Config(), {overrides!r})
    s = build(cfg)
    real_add = s.lexical.add_chunks
    def add_then_die(generation, chunks):
        real_add(generation, chunks)
        os._exit(17)  # killed mid-transaction: no rollback, no cleanup code runs
    s.lexical.add_chunks = add_then_die
    s.ingest.ingest("demo", {root!r})
    """
)


def test_killed_process_mid_staging_leaves_previous_generation(tmp_path: Path) -> None:
    cfg = _config(tmp_path, "memory")
    root = _corpus(tmp_path)
    s = build(cfg)
    s.search.register_project_root(PID, str(root))
    g1 = s.ingest.ingest(PID, str(root)).index_generation
    n1 = _counts(s, g1)
    s.close()
    _modify(root)
    overrides = {
        "lexical_store.path": cfg.lexical_store.path,
        "vector_store.provider": "memory",
        "vector_store.persist_path": cfg.vector_store.persist_path,
        "vector_store.collection": "crash_test",
    }
    src = str(Path(__file__).resolve().parents[2] / "src")
    code = CHILD.format(src=src, overrides=overrides, root=str(root))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=os.environ
    )
    assert proc.returncode == 17, proc.stderr
    s2 = build(cfg)
    s2.search.register_project_root(PID, str(root))
    assert s2.catalog.active_generation(PID) == g1
    assert s2.gc_removed, "the crashed staging must be collected at start"
    assert s2.catalog.stale_generations() == []
    assert _counts(s2, g1) == n1
    _assert_serves(s2, g1)
    s2.close()
