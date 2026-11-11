"""Download the pinned GGUF files and verify their sha256 (`make models`; build plan S3.1).

Reads configs/models.lock.yaml. Each file is streamed from
https://huggingface.co/<repo>/resolve/<revision>/<file> into <models_dir>/<file>.part, hashed while
downloading, and renamed into place only when the sha256 matches. Existing files are re-hashed
and kept when they match. Public model weights only; nothing is uploaded.

  uv run python scripts/fetch_models.py                 # all pinned models
  uv run python scripts/fetch_models.py --only qwen3-embedding-4b-q8_0 bge-reranker-v2-m3-q8_0
  uv run python scripts/fetch_models.py --verify-only   # hash what is there, download nothing
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from project_rag.adapters.models_lock import ModelArtifact, load_models_lock  # noqa: E402

CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _download(art: ModelArtifact, dest: Path) -> str:
    url = f"https://huggingface.co/{art.repo}/resolve/{art.revision}/{art.file}"
    part = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    t0 = time.monotonic()
    done = 0
    with urllib.request.urlopen(url, timeout=60) as r, part.open("wb") as out:
        while True:
            block = r.read(CHUNK)
            if not block:
                break
            out.write(block)
            h.update(block)
            done += len(block)
            if done % (256 * CHUNK) < CHUNK:
                rate = done / max(1e-6, time.monotonic() - t0) / 1e6
                print(f"  {art.file}: {done / 1e9:.2f}/{art.size / 1e9:.2f} GB ({rate:.0f} MB/s)")
    digest = h.hexdigest()
    if digest != art.sha256:
        part.unlink()
        raise SystemExit(f"SHA MISMATCH {art.file}: got {digest}, pinned {art.sha256}")
    part.replace(dest)
    return digest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", default=str(REPO_ROOT / "configs" / "models.lock.yaml"))
    ap.add_argument("--only", nargs="*", default=None, help="model keys to fetch")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    lock = load_models_lock(args.lock)
    models_dir = (REPO_ROOT / lock.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for key, art in lock.models.items():
        if args.only and key not in args.only:
            continue
        dest = models_dir / art.file
        if dest.is_file():
            digest = _sha256(dest)
            ok = digest == art.sha256
            print(f"{'OK  ' if ok else 'BAD '} {key} {art.file} sha256={digest}")
            failed |= not ok
            continue
        if args.verify_only:
            print(f"MISS {key} {art.file}")
            failed = True
            continue
        print(f"GET  {key} {art.repo}@{art.revision[:12]} {art.file}")
        print(f"OK   {key} {art.file} sha256={_download(art, dest)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
