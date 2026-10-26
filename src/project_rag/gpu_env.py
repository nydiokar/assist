"""GPU environment preparation for ROCm-on-WSL (librocdxg), see docs/DECISIONS.md 2026-10-26.

ROCm < 7.13 under WSL only sees the GPU through /dev/dxg when HSA_ENABLE_DXG_DETECTION=1.
That variable used to live only in ~/.bashrc, which non-interactive shells (`wsl -- bash -c`,
agents, cron, CI) never read, so torch.cuda.is_available() was False with no hint why.
prepare_gpu_env() makes the requirement independent of the launching shell; describe_gpu_env()
turns a bare "no CUDA" into an actionable diagnosis. Neither falls back to CPU.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

DXG_ENV: str = "HSA_ENABLE_DXG_DETECTION"
DXG_DEVICE: Path = Path("/dev/dxg")


class GpuEnvReport(BaseModel):
    is_wsl: bool
    dxg_present: bool
    dxg_env: str | None
    torch_hip: str | None
    torch_cuda: str | None


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def prepare_gpu_env() -> None:
    """Call BEFORE `import torch`. No-op outside WSL or when the GPU device node is absent."""
    if is_wsl() and DXG_DEVICE.exists():
        os.environ.setdefault(DXG_ENV, "1")


def describe_gpu_env(torch_hip: str | None, torch_cuda: str | None) -> str:
    report = GpuEnvReport(
        is_wsl=is_wsl(), dxg_present=DXG_DEVICE.exists(), dxg_env=os.environ.get(DXG_ENV),
        torch_hip=torch_hip, torch_cuda=torch_cuda)
    hint: str = ""
    if report.torch_hip and report.is_wsl and not report.dxg_present:
        hint = " /dev/dxg is missing: WSL GPU passthrough/Windows driver problem."
    elif report.torch_hip and report.is_wsl and report.dxg_env != "1":
        hint = f" {DXG_ENV}=1 is not set in this process (ROCm-on-WSL needs it)."
    elif report.torch_hip and not report.is_wsl:
        hint = " ROCm torch outside WSL: run this inside WSL (Ubuntu-24.04 ROCm venv)."
    elif not report.torch_hip and not report.torch_cuda:
        hint = " This torch build has neither CUDA nor ROCm (CPU wheel): wrong venv."
    return (f"GPU env: {report.model_dump_json()}.{hint} "
            "Do NOT restart WSL (`wsl --shutdown`) and do not conclude 'GPU down' without this report.")
