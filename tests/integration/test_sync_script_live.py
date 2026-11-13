"""S8 exit check: deploy/rclone/sync.sh in the pinned rclone image against a local-path remote
(stands in for Drive; no Google account): changes trigger an ingest, an unchanged re-run does
not, and a mass deletion trips --max-delete and leaves /corpus intact. `live`: needs Docker."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.live
REPO = Path(__file__).resolve().parents[2]
RCLONE = (
    "rclone/rclone:1.68.2@sha256:74c51b8817e5431bd6d7ed27cb2a50d8ee78d77f6807b72a41ef6f898845942b"
)
STUB = (
    "import http.server,sys\n"
    "class H(http.server.BaseHTTPRequestHandler):\n"
    "    def do_POST(self):\n"
    "        print('POST', self.path, flush=True); self.send_response(202); self.end_headers();"
    ' self.wfile.write(b\'{"ingest": "started"}\')\n'
    "http.server.HTTPServer(('0.0.0.0', 8001), H).serve_forever()\n"
)


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=300, check=check
    )


@pytest.fixture
def env(tmp_path: Path):
    if shutil.which("docker") is None:
        pytest.skip("docker not available")
    tag = uuid.uuid4().hex[:8]
    net, stub = f"synctest-{tag}", f"stub-{tag}"
    vol = f"synccorpus-{tag}"  # a named volume, as in production (bind mounts on Windows drop
    # the modtimes rclone sets, which makes every run re-copy)
    _docker("network", "create", net)
    _docker("volume", "create", vol)
    _docker(
        "run",
        "-d",
        "--name",
        stub,
        "--network",
        net,
        "python:3.12-slim-bookworm",
        "python",
        "-c",
        STUB,
    )
    src = tmp_path / "drive"
    (src / "Proj").mkdir(parents=True)
    for i in range(60):
        (src / "Proj" / f"doc_{i:02d}.txt").write_text(f"note {i}\n", encoding="utf-8")
    (src / "Proj" / "clip.mp4").write_bytes(b"\x00" * 16)  # filtered out (D-9)
    yield net, stub, src, vol
    _docker("rm", "-f", stub, check=False)
    _docker("network", "rm", net, check=False)
    _docker("volume", "rm", vol, check=False)


def _ls(vol: str, pattern: str) -> list[str]:
    out = _docker(
        "run",
        "--rm",
        "-v",
        f"{vol}:/corpus:ro",
        "--entrypoint",
        "sh",
        RCLONE,
        "-c",
        f"ls {pattern} 2>/dev/null || true",
    ).stdout
    return [ln for ln in out.split() if ln]


def _sync(net: str, stub: str, src: Path, dest: str) -> subprocess.CompletedProcess[str]:
    return _docker(
        "run",
        "--rm",
        "--network",
        net,
        "-v",
        f"{src}:/src:ro",
        "-v",
        f"{dest}:/corpus",
        "-v",
        f"{REPO / 'deploy' / 'rclone'}:/deploy/rclone:ro",
        "-e",
        "RCLONE_REMOTE=/src",
        "-e",
        f"ADMIN_URL=http://{stub}:8001",
        "--entrypoint",
        "sh",
        RCLONE,
        "/deploy/rclone/sync.sh",
        check=False,
    )


def test_sync_triggers_ingest_and_max_delete_protects(env) -> None:
    net, stub, src, dest = env
    first = _sync(net, stub, src, dest)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "SYNC OK: changed=60" in first.stdout and "INGEST TRIGGERED" in first.stdout
    assert len(_ls(dest, "/corpus/Proj/doc_*.txt")) == 60
    assert _ls(dest, "/corpus/Proj/clip.mp4") == []  # media filtered
    assert _ls(dest, "/corpus/.sync/lsjson.json") == ["/corpus/.sync/lsjson.json"]

    again = _sync(net, stub, src, dest)
    assert again.returncode == 0 and "SYNC OK: changed=0" in again.stdout
    assert "INGEST TRIGGERED" not in again.stdout

    (src / "Proj" / "doc_00.txt").write_text("edited\n", encoding="utf-8")
    edited = _sync(net, stub, src, dest)
    assert "SYNC OK: changed=1" in edited.stdout and "INGEST TRIGGERED" in edited.stdout

    for p in list((src / "Proj").glob("doc_*.txt"))[:55]:
        p.unlink()
    mass = _sync(net, stub, src, dest)
    assert mass.returncode != 0
    assert "max-delete threshold (50) reached" in mass.stdout
    assert len(_ls(dest, "/corpus/Proj/doc_*.txt")) >= 10  # at most 50 deletions happened
    logs = _docker("logs", stub).stdout
    assert logs.count("POST /admin/ingest") == 2
