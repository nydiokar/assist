"""Filesystem artifact store for the parse/chunk stage caches (A01 §C).

Layout: <root>/_parse_shared/<key>.json (cross-project: identical bytes parse identically) and
<root>/<project_id>/chunk/<key>.json (per project: chunk identity carries project lineage).
Plain JSON, Pydantic round-trip.
"""

from __future__ import annotations

from pathlib import Path

from ..domain.models import ChunkArtifact, ParseArtifact


class FsArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _parse_path(self, key: str) -> Path:
        return self._root / "_parse_shared" / f"{key}.json"

    def _chunk_path(self, project_id: str, key: str) -> Path:
        return self._root / project_id / "chunk" / f"{key}.json"

    def get_parse(self, key: str) -> ParseArtifact | None:
        p = self._parse_path(key)
        if not p.is_file():
            return None
        return ParseArtifact.model_validate_json(p.read_text(encoding="utf-8"))

    def put_parse(self, artifact: ParseArtifact) -> None:
        p = self._parse_path(artifact.artifact_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(artifact.model_dump_json(), encoding="utf-8")

    def get_chunk(self, project_id: str, key: str) -> ChunkArtifact | None:
        p = self._chunk_path(project_id, key)
        if not p.is_file():
            return None
        return ChunkArtifact.model_validate_json(p.read_text(encoding="utf-8"))

    def put_chunk(self, project_id: str, artifact: ChunkArtifact) -> None:
        p = self._chunk_path(project_id, artifact.artifact_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(artifact.model_dump_json(), encoding="utf-8")
