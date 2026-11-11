"""Project registry (S4, spec §6.4): one project per top-level folder of the corpus root.

The registry key is the Drive folder id from the sync sidecar (`.sync/manifest.json`, written by
the S8 sync job), which is stable across renames: a renamed folder keeps its `project_id` (a slug
assigned once) and only `display_name` follows the new name, so nothing is re-indexed (source
paths are relative to the project folder). Without a sidecar (dev), the key falls back to a hash
of the folder name, documented as not rename-stable.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import ValidationError

from ..core.slug import fallback_drive_key, unique_slug
from ..domain.models import IngestReport, ProjectIdentity, ProjectRecord, SyncManifest
from ..ports.catalog import Catalog
from ..ports.corpus import CorpusReader

SIDECAR = ".sync/manifest.json"


class ProjectRegistryService:
    def __init__(self, catalog: Catalog, corpus: CorpusReader) -> None:
        self._catalog = catalog
        self._corpus = corpus

    def read_manifest(self, corpus_root: str) -> SyncManifest | None:
        raw = self._corpus.read_text(corpus_root, SIDECAR)
        if raw is None:
            return None
        try:
            return SyncManifest.model_validate_json(raw)
        except ValidationError:
            return None  # a corrupt sidecar falls back to path keys; the run report says so

    def sync(self, corpus_root: str) -> list[ProjectRecord]:
        """Register every top-level folder; renames update display_name only."""
        manifest = self.read_manifest(corpus_root)
        folder_ids = {f.path: f.drive_id for f in manifest.folders} if manifest else {}
        taken = self._catalog.all_project_ids()
        records: list[ProjectRecord] = []
        for folder in self._corpus.list_dirs(corpus_root):
            drive_id = folder_ids.get(folder)
            key_source = "drive" if drive_id else "path-fallback"
            key = drive_id or fallback_drive_key(folder)
            known = self._catalog.project_by_drive_id(key)
            renamed_from: str | None = None
            if known is None:
                identity = ProjectIdentity(
                    project_id=unique_slug(folder, taken), drive_id=key, display_name=folder
                )
                taken = taken | {identity.project_id}
                self._catalog.upsert_project_identity(identity)
            elif known.display_name != folder:
                renamed_from = known.display_name
                identity = known.model_copy(update={"display_name": folder})
                self._catalog.upsert_project_identity(identity)
            else:
                identity = known
            records.append(
                ProjectRecord(
                    identity=identity,
                    folder=folder,
                    root_path=self._corpus.resolve(corpus_root, folder),
                    key_source=key_source,
                    renamed_from=renamed_from,
                )
            )
        return records

    def project_ids(self) -> set[str]:
        return {p.project_id for p in self._catalog.list_projects()}


class Ingests(Protocol):
    def ingest(self, project_id: str, root_path: str, *, dry_run: bool = False) -> IngestReport: ...


def ingest_corpus(
    registry: ProjectRegistryService, ingest: Ingests, corpus_root: str
) -> list[tuple[ProjectRecord, IngestReport]]:
    """Sync the registry, then ingest every project under its stable project_id."""
    return [
        (rec, ingest.ingest(rec.identity.project_id, rec.root_path))
        for rec in registry.sync(corpus_root)
    ]
