"""Start-up model check (spec §6.7/§8.1): every served GGUF the profile names must be present
and match its pinned sha256. Reads the shared /models volume read-only."""

from __future__ import annotations

from pathlib import Path

from ..domain.config import Config
from .models_lock import file_sha256


def gguf_problems(cfg: Config, models_dir: str | Path) -> list[str]:
    problems: list[str] = []
    checks: list[tuple[str, str, str]] = []
    if cfg.embedding.provider == "openai_compat":
        o = cfg.embedding.openai_compat
        checks.append(("embedding", o.gguf_file, o.gguf_sha256))
    r = cfg.retrieval.rerank
    if r.enabled and r.provider == "openai_compat":
        checks.append(("rerank", r.openai_compat.gguf_file, r.openai_compat.gguf_sha256))
    for role, name, pinned in checks:
        if not name or not pinned:
            problems.append(f"{role}: gguf_file/gguf_sha256 not pinned")
            continue
        path = Path(models_dir) / name
        if not path.is_file():
            problems.append(f"{role}: model file missing: {path}")
            continue
        actual = file_sha256(path)
        if actual != pinned:
            problems.append(f"{role}: sha256 mismatch for {name}: {actual} != pinned {pinned}")
    return problems
