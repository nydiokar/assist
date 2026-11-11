"""Cross-language query expansion logic (F11/F14, Task B). Pure, offline, deterministic.

Tests the mechanism contract only — the real-corpus measurement (a NEGATIVE: net evidence@5
0.619->0.595, yp09 did not move, yp02 regressed) is in EVAL_LOG 2026-10-28. The feature ships
OFF by default for that reason; these tests pin its behavior so the default-off capability stays
correct and reversible.
"""

from __future__ import annotations

from project_rag.core.query_expansion import expand_query


def test_same_script_query_unchanged():
    # Bulgarian query over a Bulgarian corpus: no EN glosses appended.
    q = "Колко уебинара са проведени?"
    assert expand_query(q, corpus_script="cyrillic") == q


def test_latin_query_gets_bg_anchors_appended():
    out = expand_query("How many webinars were planned and achieved?", corpus_script="cyrillic")
    assert out.startswith("How many webinars were planned and achieved?")  # original preserved
    assert "уебинари" in out and "заложени" in out and "постигнати" in out


def test_multiword_term_matched_before_single_words():
    out = expand_query("How many decision makers attended?", corpus_script="cyrillic")
    assert "Вземащи решения" in out


def test_no_matching_terms_returns_original():
    q = "What color is the logo?"
    assert expand_query(q, corpus_script="cyrillic") == q


def test_glosses_are_deduplicated():
    # "planned" and "target" both map to "заложени" — it must appear once.
    out = expand_query("What is the planned target?", corpus_script="cyrillic")
    assert out.count("заложени") == 1


def test_neutral_query_unchanged():
    assert expand_query("12345", corpus_script="cyrillic") == "12345"
