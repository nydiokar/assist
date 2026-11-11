"""Compatibility shim (S1 move in progress): models live in `project_rag.domain.config`."""

from __future__ import annotations

from pathlib import Path

import yaml

from .domain.config import *  # noqa: F403
from .domain.config import Config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "retrieval.yaml"


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Config.model_validate(raw)
