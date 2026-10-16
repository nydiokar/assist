"""Discovery must handle paths beyond Windows MAX_PATH (real corpus has 277-char paths)."""

from __future__ import annotations

import os
from pathlib import Path

from project_rag.discovery import discover


def test_discover_path_longer_than_260(tmp_path: Path) -> None:
    prefix: str = "\\\\?\\" if os.name == "nt" else ""
    deep: str = os.path.join(str(tmp_path), *["д" * 40] * 7)
    os.makedirs(prefix + deep)
    target: str = os.path.join(prefix + deep, "доклад.txt")
    with open(target, "w", encoding="utf-8") as f:
        f.write("текст")
    assert len(deep) > 260

    found = discover(tmp_path)
    assert [d.ext for d in found] == [".txt"]
    assert found[0].source_path.endswith("доклад.txt")
    assert "\\" not in found[0].source_path
    assert found[0].size > 0 and len(found[0].content_hash) == 64
