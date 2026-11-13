"""Canonical choice for exact duplicates (S5): deterministic, order-independent, preference rules."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from project_rag.core.ingest_policy import exact_duplicates
from project_rag.domain.models import DiscoveredFile


def _f(path: str, h: str) -> DiscoveredFile:
    return DiscoveredFile(
        source_path=path,
        abs_path="/x/" + path,
        ext=".pdf",
        size=1,
        mtime=0.0,
        content_hash=h,
        supported=True,
    )


def test_preference_rules():
    files = [
        _f("Res/Guide.pdf", "a"),
        _f("Meeting/Pack/Guide.pdf", "a"),
        _f("Meeting/Copy of Guide.pdf", "a"),
        _f("Meeting/Guide (1).pdf", "a"),
        _f("solo.pdf", "b"),
    ]
    dup = exact_duplicates(files, ("Res/",))
    # "Copy of"/"(1)" lose; reference loses to project record even when deeper.
    assert set(dup) == {"Res/Guide.pdf", "Meeting/Copy of Guide.pdf", "Meeting/Guide (1).pdf"}
    assert set(dup.values()) == {"Meeting/Pack/Guide.pdf"}
    assert "solo.pdf" not in dup


def test_cyrillic_copy_name_and_no_duplicates():
    files = [_f("A/Копие на ПРАВИЛА.pdf", "a"), _f("A/Z/ПРАВИЛА.pdf", "a")]
    assert exact_duplicates(files) == {"A/Копие на ПРАВИЛА.pdf": "A/Z/ПРАВИЛА.pdf"}
    assert exact_duplicates([_f("a.pdf", "1"), _f("b.pdf", "2")]) == {}


_paths = st.lists(
    st.tuples(st.sampled_from(["a", "b", "c"]), st.text("xyz/ ", min_size=1, max_size=6)),
    min_size=1,
    max_size=12,
    unique_by=lambda t: t[1],
)


@given(_paths, st.randoms())
def test_order_independent_and_partitions_groups(items, rnd):
    files = [_f(p, h) for h, p in items]
    shuffled = list(files)
    rnd.shuffle(shuffled)
    dup = exact_duplicates(files)
    assert dup == exact_duplicates(shuffled)
    by_hash: dict[str, list[str]] = {}
    for f in files:
        by_hash.setdefault(f.content_hash, []).append(f.source_path)
    # Exactly one survivor per hash group; every duplicate points at its group's survivor.
    survivors = {p for ps in by_hash.values() for p in ps if p not in dup}
    assert len(survivors) == len(by_hash)
    for p, canon in dup.items():
        assert canon in survivors and canon != p
