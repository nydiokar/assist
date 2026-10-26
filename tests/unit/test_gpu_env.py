import os
from pathlib import Path

import pytest

from project_rag import gpu_env


def test_prepare_sets_var_on_wsl_with_dxg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dxg: Path = tmp_path / "dxg"
    dxg.touch()
    monkeypatch.setattr(gpu_env, "is_wsl", lambda: True)
    monkeypatch.setattr(gpu_env, "DXG_DEVICE", dxg)
    monkeypatch.delenv(gpu_env.DXG_ENV, raising=False)
    gpu_env.prepare_gpu_env()
    assert os.environ[gpu_env.DXG_ENV] == "1"


def test_prepare_respects_existing_value(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dxg: Path = tmp_path / "dxg"
    dxg.touch()
    monkeypatch.setattr(gpu_env, "is_wsl", lambda: True)
    monkeypatch.setattr(gpu_env, "DXG_DEVICE", dxg)
    monkeypatch.setenv(gpu_env.DXG_ENV, "0")
    gpu_env.prepare_gpu_env()
    assert os.environ[gpu_env.DXG_ENV] == "0"


def test_prepare_noop_off_wsl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_env, "is_wsl", lambda: False)
    monkeypatch.delenv(gpu_env.DXG_ENV, raising=False)
    gpu_env.prepare_gpu_env()
    assert gpu_env.DXG_ENV not in os.environ


def test_describe_names_missing_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dxg: Path = tmp_path / "dxg"
    dxg.touch()
    monkeypatch.setattr(gpu_env, "is_wsl", lambda: True)
    monkeypatch.setattr(gpu_env, "DXG_DEVICE", dxg)
    monkeypatch.delenv(gpu_env.DXG_ENV, raising=False)
    msg: str = gpu_env.describe_gpu_env("7.2.4", None)
    assert "HSA_ENABLE_DXG_DETECTION=1 is not set" in msg
    assert "wsl --shutdown" in msg


def test_describe_flags_cpu_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_env, "is_wsl", lambda: False)
    assert "CPU wheel" in gpu_env.describe_gpu_env(None, None)
