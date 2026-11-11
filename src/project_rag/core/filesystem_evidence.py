"""Filesystem-presence evidence for unparseable files (owner-requested 2026-10-28; DECISIONS).

The corpus holds 162 files the pipeline cannot parse (images, video, audio, legacy .doc) — a known
coverage gap (OCR/transcription are out of scope). Their *filenames and containing folders*,
however, carry meaning: Bulgarian event/topic names like `Уебинари/2. Младежко участие на практика`
("Webinars / 2. Youth participation in practice"). Discovery already walks every file; this module
turns that walk into lightweight, clearly-labelled synthetic chunks so a question like "did Webinar
2 happen / what was it about?" can surface the named, non-empty folder as *citable circumstantial
evidence* — never mistaken for confirmed document content.

Design (DECISIONS 2026-10-28):
- Granularity is **per folder that contains at least one unparseable file**, not per file: most
  image filenames are opaque hashes (`506535769_1127905…n.jpg`) that would only pollute retrieval;
  the signal is the folder name + that the folder is non-empty. Parseable files already have real
  content chunks, so folders with only parseable files are NOT synthesized here.
- Each chunk is tagged `doc_class="filesystem"` so it is filterable at query time and the future LLM
  can never confuse presence with content, and its text opens with an explicit
  "[FILESYSTEM PRESENCE — NOT DOCUMENT CONTENT]" banner.
- chunk/source/version ids are deterministic (v1 §4.3 discipline) so re-ingesting an unchanged tree
  yields identical chunks; the version id hashes the folder's file inventory so adding/removing a
  file changes it.
"""

from __future__ import annotations

import hashlib
import posixpath
from collections import defaultdict

from ..domain.models import BlockType, Chunk, DiscoveredFile, SourceLocation

FILESYSTEM_DOC_CLASS = "filesystem"
FILESYSTEM_VERSION = "fs_presence_v1"
_BANNER = "[FILESYSTEM PRESENCE — NOT DOCUMENT CONTENT]"
_MAX_NAMES = 12  # cap representative filenames per folder so a 53-image folder stays a short chunk


def _folder_of(source_path: str) -> str:
    # Pure string op (no filesystem access): discovery emits normalised forward-slash paths.
    return posixpath.dirname(source_path)


def _fs_source_id(folder: str) -> str:
    return "fs_" + hashlib.sha256(("FS|" + folder).encode("utf-8")).hexdigest()[:20]


def _fs_version_id(folder: str, inventory: list[str]) -> str:
    raw = folder + "|" + FILESYSTEM_VERSION + "|" + "|".join(sorted(inventory))
    return "fv_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _fs_chunk_id(project_id: str, source_id: str, version_id: str) -> str:
    raw = f"{project_id}|{source_id}|{version_id}|{FILESYSTEM_VERSION}"
    return "ckfs_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_filesystem_chunks(project_id: str, discovered: list[DiscoveredFile]) -> list[Chunk]:
    """One presence chunk per folder that contains >=1 unparseable file.

    The chunk text names the folder, counts parseable vs unparseable files (with an extension
    breakdown), lists representative unparseable filenames (capped), and is prefixed with an
    explicit not-content banner. Deterministic; returns chunks sorted by folder for stable ids.
    """
    by_folder: dict[str, list[DiscoveredFile]] = defaultdict(list)
    for d in discovered:
        by_folder[_folder_of(d.source_path)].append(d)

    chunks: list[Chunk] = []
    for folder in sorted(by_folder):
        files = by_folder[folder]
        unsupported = [f for f in files if not f.supported]
        if not unsupported:
            continue  # parseable-only folder is already represented by content chunks
        ext_counts: dict[str, int] = defaultdict(int)
        for f in unsupported:
            ext_counts[f.ext or "(none)"] += 1
        ext_summary = ", ".join(f"{ext}×{n}" for ext, n in sorted(ext_counts.items()))
        names = [posixpath.basename(f.source_path) for f in unsupported]
        shown = names[:_MAX_NAMES]
        names_line = "; ".join(shown) + ("; …" if len(names) > _MAX_NAMES else "")
        folder_label = folder or "(project root)"

        body = (
            f"{_BANNER}\n"
            f"Folder: {folder_label}\n"
            f"The folder contains {len(files)} file(s): {len(unsupported)} unparseable "
            f"({ext_summary}) and {len(files) - len(unsupported)} parseable.\n"
            f"Unparseable file names (not transcribed/OCR'd): {names_line}\n"
            f"This chunk records only that these files exist at this path. Their contents are "
            f"NOT indexed (no OCR, no transcription). Use as circumstantial evidence that an "
            f"activity or material exists, never as confirmation of what the files say."
        )

        source_id = _fs_source_id(folder)
        inventory = [f.source_path for f in files]
        version_id = _fs_version_id(folder, inventory)
        chunk_id = _fs_chunk_id(project_id, source_id, version_id)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                project_id=project_id,
                source_id=source_id,
                version_id=version_id,
                document_id=source_id,
                embed_text=body,
                citation_text=body,
                block_type=BlockType.PARAGRAPH,
                heading=folder_label,
                location=SourceLocation(block_path=folder_label),
                chunk_ordinal=0,
                source_path=folder_label,
                source_type="filesystem",
                doc_class=FILESYSTEM_DOC_CLASS,
                token_count=len(body.split()),
                truncated=False,
            )
        )
    return chunks
