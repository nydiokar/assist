"""Legacy Word `.doc` via antiword (D-10). Content problems become document states, not raises.

`antiword -m UTF-8.txt -w 0 <file>` prints the text as UTF-8 with no line wrapping; blank lines
separate paragraphs. A missing binary, a non-zero exit or a timeout yields FAILED with a reason.
"""

from __future__ import annotations

import shutil
import subprocess

from ..domain.models import Block, BlockType, DocState, ParsedDocument, SourceLocation


class AntiwordParser:
    def __init__(self, *, binary: str = "antiword", timeout_s: float = 60.0) -> None:
        self._binary = binary
        self._timeout_s = timeout_s

    def can_parse(self, ext: str) -> bool:
        return ext == ".doc"

    def parse(self, abs_path: str, source_path: str) -> ParsedDocument:
        exe = shutil.which(self._binary)
        if exe is None:
            return _failed(source_path, f"antiword not installed ({self._binary})")
        try:
            proc = subprocess.run(
                [exe, "-m", "UTF-8.txt", "-w", "0", abs_path],
                capture_output=True,
                timeout=self._timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _failed(source_path, f"antiword timed out after {self._timeout_s:.0f}s")
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()[:300]
            return _failed(source_path, f"antiword exit {proc.returncode}: {err}")
        text = proc.stdout.decode("utf-8", errors="replace")
        paragraphs = [p.strip() for p in _split_blank(text) if p.strip()]
        if not paragraphs:
            return ParsedDocument(
                source_path=source_path,
                state=DocState.NO_TEXT,
                diagnostics=["antiword produced no text"],
            )
        blocks = [
            Block(
                block_type=BlockType.PARAGRAPH,
                text=p,
                location=SourceLocation(block_path=f"p{i}"),
            )
            for i, p in enumerate(paragraphs)
        ]
        return ParsedDocument(
            source_path=source_path,
            state=DocState.PARSED,
            blocks=blocks,
            diagnostics=["parser=antiword"],
        )


def _split_blank(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if line.strip():
            buf.append(" ".join(line.split()))
        elif buf:
            parts.append(" ".join(buf))
            buf = []
    if buf:
        parts.append(" ".join(buf))
    return parts


def _failed(source_path: str, reason: str) -> ParsedDocument:
    return ParsedDocument(source_path=source_path, state=DocState.FAILED, diagnostics=[reason])
