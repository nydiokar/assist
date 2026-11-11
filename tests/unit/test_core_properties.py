"""Property tests for the pure core (v3 build plan S1.2; spec A03 §15.3 core list)."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from project_rag.core.cap import apply_per_file_cap
from project_rag.core.chunking import chunk_document
from project_rag.core.corpus_diff import diff_corpus
from project_rag.core.fusion import FusedCandidate, rrf, rrf_fuse
from project_rag.domain.config import ChunkingConfig
from project_rag.domain.models import (
    Block,
    BlockType,
    ChannelHit,
    DiscoveredFile,
    DocState,
    HitPayload,
    ParsedDocument,
    SourceLocation,
)

# ---- fusion ----

_ids = st.lists(st.sampled_from([f"ck_{i}" for i in range(12)]), unique=True, max_size=12)


def _hits(ids: list[str]) -> list[ChannelHit]:
    payload = HitPayload(document_id="d", source_path="p.md")
    return [ChannelHit(chunk_id=c, score=float(i), payload=payload) for i, c in enumerate(ids)]


def _order(cands: list[FusedCandidate]) -> list[tuple[str, float]]:
    return [(c.chunk_id, c.fused) for c in cands]


@given(dense=_ids, lexical=_ids, k=st.integers(min_value=1, max_value=200))
def test_rrf_fuse_is_invariant_to_channel_order(
    dense: list[str], lexical: list[str], k: int
) -> None:
    a = rrf_fuse({"dense": _hits(dense), "lexical": _hits(lexical)}, k)
    b = rrf_fuse({"lexical": _hits(lexical), "dense": _hits(dense)}, k)
    assert _order(a) == _order(b)


@given(dense=_ids, lexical=_ids, k=st.integers(min_value=1, max_value=200))
def test_rrf_fuse_sorted_and_scores_are_rank_sums(
    dense: list[str], lexical: list[str], k: int
) -> None:
    fused = rrf_fuse({"dense": _hits(dense), "lexical": _hits(lexical)}, k)
    assert {c.chunk_id for c in fused} == set(dense) | set(lexical)
    keys = [(-c.fused, c.chunk_id) for c in fused]
    assert keys == sorted(keys)
    for c in fused:
        expected = sum(rrf(r, k) for r in c.ranks.values())
        assert c.fused == expected


@given(
    rank=st.integers(min_value=1, max_value=500),
    k=st.integers(min_value=1, max_value=500),
    dk=st.integers(min_value=1, max_value=100),
)
def test_rrf_is_monotone_in_k_and_rank(rank: int, k: int, dk: int) -> None:
    assert rrf(rank, k + dk) < rrf(rank, k)
    assert rrf(rank + dk, k) < rrf(rank, k)


# ---- per-file cap ----

_paths = st.sampled_from(["a.md", "b.md", "c.md", "d.md"])


@st.composite
def _ranking(draw: st.DrawFn) -> list[FusedCandidate]:
    paths = draw(st.lists(_paths, max_size=25))
    return [
        FusedCandidate(
            chunk_id=f"ck_{i:02d}",
            payload=HitPayload(document_id="d", source_path=p),
            ranks={"dense": i + 1},
            fused=1.0 / (i + 1),
        )
        for i, p in enumerate(paths)
    ]


@given(
    ranking=_ranking(),
    per_file=st.integers(min_value=1, max_value=5),
    window=st.integers(min_value=1, max_value=20),
)
def test_cap_order_length_and_refill_rule(
    ranking: list[FusedCandidate], per_file: int, window: int
) -> None:
    picked = apply_per_file_cap(ranking, per_file, window)
    ids = [c.chunk_id for c in picked]
    # Length: never short when candidates exist; never longer than the window.
    assert len(picked) == min(window, len(ranking))
    # No duplicates; every pick comes from the ranking.
    assert len(set(ids)) == len(ids)
    assert set(ids) <= {c.chunk_id for c in ranking}
    # Admitted-under-cap picks keep rank order and precede any refill pick.
    position = {c.chunk_id: i for i, c in enumerate(ranking)}
    counts: dict[str, int] = {}
    first_part: list[str] = []
    for c in picked:
        n = counts.get(c.payload.source_path, 0)
        if n >= per_file:
            break
        counts[c.payload.source_path] = n + 1
        first_part.append(c.chunk_id)
    assert [position[i] for i in first_part] == sorted(position[i] for i in first_part)
    # A file exceeds the cap only as refill, i.e. when every under-cap candidate is already in.
    per_path: dict[str, int] = {}
    for c in picked:
        per_path[c.payload.source_path] = per_path.get(c.payload.source_path, 0) + 1
    if any(n > per_file for n in per_path.values()):
        admissible: dict[str, int] = {}
        for c in ranking:
            if admissible.get(c.payload.source_path, 0) < per_file:
                admissible[c.payload.source_path] = admissible.get(c.payload.source_path, 0) + 1
                assert c.chunk_id in ids


# ---- chunking ----

_word = st.text(alphabet="abcdefghijklmnopqrstuvwxyzабвгдежзий", min_size=1, max_size=12)


@st.composite
def _document(draw: st.DrawFn) -> ParsedDocument:
    n = draw(st.integers(min_value=0, max_value=12))
    blocks: list[Block] = []
    for i in range(n):
        if draw(st.booleans()) and i % 4 == 0:
            blocks.append(
                Block(
                    block_type=BlockType.HEADING,
                    text=f"Heading {i}",
                    location=SourceLocation(block_path=f"h{i}"),
                )
            )
            continue
        words = draw(st.lists(_word, min_size=1, max_size=300))
        # A unique marker per block keeps the boilerplate filter (repeated short paragraphs) off.
        text = f"blk{i} " + " ".join(words)
        blocks.append(
            Block(
                block_type=BlockType.PARAGRAPH,
                text=text,
                location=SourceLocation(block_path=f"p{i}"),
            )
        )
    return ParsedDocument(source_path="doc.md", state=DocState.PARSED, blocks=blocks)


@settings(max_examples=60, deadline=None)
@given(doc=_document(), target=st.integers(min_value=20, max_value=200))
def test_chunks_cover_input_and_respect_max_tokens(doc: ParsedDocument, target: int) -> None:
    cfg = ChunkingConfig(target_tokens=target, max_tokens=target * 4, overlap_tokens=0)
    chunks = chunk_document(
        doc, project_id="p", source_id="s", version_id="v", source_type="md", cfg=cfg
    )
    cited = " ".join(c.citation_text for c in chunks).split()
    expected = [
        w for b in doc.blocks if b.block_type == BlockType.PARAGRAPH for w in b.text.split()
    ]
    # Coverage: every paragraph word appears, in document order (split pieces keep all words).
    assert cited == expected
    for c in chunks:
        assert not c.truncated
        assert c.token_count <= cfg.max_tokens
    assert len({c.chunk_id for c in chunks}) == len(chunks)


# ---- corpus diff ----

_file_names = st.sampled_from([f"f{i}.md" for i in range(10)])
_hash = st.sampled_from(["h1", "h2", "h3"])


@given(
    prior=st.dictionaries(_file_names, _hash, max_size=10),
    now=st.dictionaries(_file_names, _hash, max_size=10),
)
def test_diff_corpus_partitions_the_union(prior: dict[str, str], now: dict[str, str]) -> None:
    discovered = [
        DiscoveredFile(
            source_path=p,
            abs_path=p,
            ext=".md",
            size=1,
            mtime=0.0,
            content_hash=h,
            supported=True,
        )
        for p, h in sorted(now.items())
    ]
    cs = diff_corpus({p: (h, "sid") for p, h in prior.items()}, discovered)
    parts = [set(cs.added), set(cs.modified), set(cs.unchanged), set(cs.deleted)]
    union: set[str] = set().union(*parts)
    assert union == set(prior) | set(now)
    assert sum(len(p) for p in parts) == len(union)  # pairwise disjoint
    assert set(cs.added) == set(now) - set(prior)
    assert set(cs.deleted) == set(prior) - set(now)
    assert set(cs.unchanged) == {p for p in now if p in prior and prior[p] == now[p]}
