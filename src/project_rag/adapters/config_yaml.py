"""Load the validated, immutable Config from YAML (the only config file reader, R7)."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..domain.config import Config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "retrieval.yaml"


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Config.model_validate(raw)
