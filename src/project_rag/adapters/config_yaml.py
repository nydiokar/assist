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


SECRET_ENV: dict[str, str] = {
    # env var -> dotted config key (spec §6.7: secrets only from the environment, never YAML)
    "ASSIST_HOSTED_EMBED_API_KEY": "embedding.openai_compat.api_key",
}


def _deep_merge(base: dict[str, object], overlay: dict[str, object]) -> dict[str, object]:
    out = dict(base)
    for key, value in overlay.items():
        prior = out.get(key)
        if isinstance(prior, dict) and isinstance(value, dict):
            out[key] = _deep_merge(prior, value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def load_layered(
    base: str | Path, overlay: str | Path | None = None, env: dict[str, str] | None = None
) -> Config:
    """configs/base.yaml (in the image) + /config/deployment.yaml (per host) + environment secrets
    (spec §6.7). A secret found in a YAML file is rejected: secrets live in the environment."""
    raw: dict[str, object] = yaml.safe_load(Path(base).read_text(encoding="utf-8")) or {}
    if overlay is not None and Path(overlay).is_file():
        over = yaml.safe_load(Path(overlay).read_text(encoding="utf-8")) or {}
        raw = _deep_merge(raw, over)
    cfg = Config.model_validate(raw)
    if cfg.embedding.openai_compat.api_key:
        raise ValueError("embedding.openai_compat.api_key must come from the environment, not YAML")
    secrets = {
        key: value for name, key in SECRET_ENV.items() if (value := (env or {}).get(name, ""))
    }
    if secrets:
        from ..domain.config import with_overrides

        cfg = with_overrides(cfg, secrets)
    return cfg
