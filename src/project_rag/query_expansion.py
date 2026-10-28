"""Cross-language query expansion (F11/F14; spec v1 §4.7 "query language mismatch").

The corpus is Bulgarian-dominant; some answers live in short BG table rows whose column headers
(`брой заложени` = planned, `Брой постигнати` = achieved, `Вземащи решения` = decision-makers)
have NO English token an EN query can match — neither lexically (FTS5 has no translation) nor
densely (a short numeric BG row is a weak dense anchor for an EN sentence). Measured: yp09
("how many decision makers…") misses at every depth from an EN query, but reaches rank 1 when the
BG anchor terms are appended (`scripts/probe_crosslang.py`, 2026-10-28).

The cheap, no-egress fix is NOT translation (that would need a hosted API — forbidden, v1 §8) and
NOT a bigger/fine-tuned embedder (F11 disposition: do not swap the model). It is a small, fixed,
local **domain glossary**: when a query is in the minority script (Latin over a Cyrillic corpus,
or vice versa), append the counterpart-language glosses of any domain terms it contains. This adds
the missing anchors to both channels before retrieval. Additive and reversible: the original query
text is preserved; expansion only appends.

Honest limitation (recorded, not hidden): this only helps terms IN the glossary. It is a curated
lexicon of the recurring project/table vocabulary, not general MT. Terms outside it are unaffected.
"""

from __future__ import annotations

import re

# EN term (lowercased, word-boundary matched) -> BG anchor(s) present in the corpus tables/text.
# Curated from the measured misses (F11/F14) and the Overview_Debate / proposal vocabulary.
_EN_TO_BG: dict[str, str] = {
    "planned": "заложени",
    "achieved": "постигнати",
    "target": "заложени",
    "actual": "постигнати",
    "decision makers": "Вземащи решения",
    "decision-makers": "Вземащи решения",
    "webinar": "уебинар",
    "webinars": "уебинари",
    "participants": "участници",
    "advocacy": "застъпнически",
    "campaign": "кампании",
    "campaigns": "кампании",
    "national meeting": "Национална среща",
    "debate": "дебат",
    "podcast": "подкаст",
    "declaration": "декларация",
    "budget": "бюджет",
    "indicator": "индикатор",
    "indicators": "индикатори",
    "municipalities": "общини",
    "institutions": "институции",
    "youth": "младежи",
}
# BG -> EN for the reverse direction (BG query over EN-bearing content, e.g. the application form).
_BG_TO_EN: dict[str, str] = {
    "бюджет": "budget",
    "кандидат": "applicant",
    "партньор": "partner",
    "индикатор": "indicator",
    "цел": "target objective",
}

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")


def _dominant_script(text: str) -> str:
    cyr = len(_CYRILLIC.findall(text))
    lat = len(_LATIN.findall(text))
    if cyr == 0 and lat == 0:
        return "neutral"
    return "cyrillic" if cyr >= lat else "latin"


def _apply_glossary(query_lower: str, glossary: dict[str, str]) -> list[str]:
    """Return the ordered, de-duplicated glosses for every glossary key found in the query.

    Multi-word keys are matched first (so "decision makers" is caught before the single words).
    """
    added: list[str] = []
    seen: set[str] = set()
    for term in sorted(glossary, key=lambda t: (-len(t.split()), t)):
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        if re.search(pattern, query_lower):
            gloss = glossary[term]
            if gloss not in seen:
                added.append(gloss)
                seen.add(gloss)
    return added


def expand_query(query: str, *, corpus_script: str = "cyrillic") -> str:
    """Append counterpart-language domain glosses when the query is in the minority script.

    `corpus_script` names the dominant script of the indexed corpus ("cyrillic" for yspdb). A Latin
    query over a Cyrillic corpus gets BG glosses appended; a Cyrillic query over a Latin corpus (or
    for the EN-bearing application form) gets EN glosses. Same-script queries are returned unchanged.
    Deterministic and additive — the original query is always the prefix of the result.
    """
    script = _dominant_script(query)
    if script == "neutral":
        return query
    ql = query.lower()
    if corpus_script == "cyrillic" and script == "latin":
        added = _apply_glossary(ql, _EN_TO_BG)
    elif corpus_script == "cyrillic" and script == "cyrillic":
        # a BG query may still need EN anchors for the application form (mixed-language corpus)
        added = _apply_glossary(ql, _BG_TO_EN)
    else:
        added = []
    if not added:
        return query
    return query + " " + " ".join(added)
