"""Pinned model artifacts (configs/models.lock.yaml): typed loader + sha256 verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class ModelArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str  # embed | rerank
    repo: str
    revision: str
    file: str
    quant: str
    size: int
    sha256: str


class ModelsLock(BaseModel):
    model_config = ConfigDict(frozen=True)

    models_dir: str
    server_image: str
    models: dict[str, ModelArtifact]


def load_models_lock(path: str | Path) -> ModelsLock:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ModelsLock.model_validate(raw)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
