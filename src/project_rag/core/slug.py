"""Project slugs and the dev-fallback registry key (S4, spec §6.4). Pure."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection

# Bulgarian streamlined transliteration (official system, 2009), lower case.
_BG = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sht",
    "ъ": "a", "ь": "y", "ю": "yu", "я": "ya",
}  # fmt: skip


def slugify(name: str, *, max_len: int = 48) -> str:
    """ASCII, lower-case, hyphenated slug; Cyrillic is transliterated. Never empty."""
    lowered = name.strip().lower()
    latin = "".join(_BG.get(ch, ch) for ch in lowered)
    slug = re.sub(r"[^a-z0-9]+", "-", latin).strip("-")[:max_len].strip("-")
    return slug or "project"


def unique_slug(name: str, taken: Collection[str]) -> str:
    base = slugify(name)
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def fallback_drive_key(folder: str) -> str:
    """Dev-only stand-in for a Drive folder id when no sync sidecar exists: a hash of the folder
    name. A rename then looks like a new project; only the sidecar's real id survives renames."""
    return "path:" + hashlib.sha256(folder.encode("utf-8")).hexdigest()[:16]
