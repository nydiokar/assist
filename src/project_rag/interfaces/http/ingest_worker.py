"""Single-writer ingest worker (spec §6.3): one ingest at a time per host, on its own thread with
its own store connection. A trigger during a run sets a rerun flag (triggers never queue up
unboundedly); the worker repeats once more when the run ends. Readers keep serving the previous
generation until the atomic pointer flip.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol

from ...domain.models import IngestReport


class Writer(Protocol):
    """What the worker needs from the writer-side services (built on the worker thread)."""

    def ingest_corpus(self, corpus_root: str, project_id: str | None) -> list[IngestReport]: ...

    def close(self) -> None: ...


class IngestWorker:
    def __init__(self, open_writer: Callable[[], Writer], corpus_root: str) -> None:
        self._open_writer = open_writer
        self._corpus_root = corpus_root
        self._lock = threading.Lock()
        self._running = False
        self._rerun = False
        self._project: str | None = None
        self._thread: threading.Thread | None = None
        self.last_reports: list[IngestReport] = []
        self.last_error: str | None = None
        self.runs = 0

    @property
    def running(self) -> bool:
        return self._running

    def trigger(self, project_id: str | None = None) -> str:
        """Start an ingest, or flag a rerun if one is in progress. Returns "started"|"rerun"."""
        with self._lock:
            if self._running:
                self._rerun = True
                if project_id is None or self._project != project_id:
                    self._project = None  # different targets collapse into one full run
                return "rerun"
            self._running = True
            self._project = project_id
            self._thread = threading.Thread(target=self._loop, name="assist-ingest", daemon=True)
            self._thread.start()
            return "started"

    def join(self, timeout: float | None = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)

    def _loop(self) -> None:
        try:
            writer = self._open_writer()  # thread-affine connection, created on this thread
        except Exception as e:  # noqa: BLE001 - service boundary: report, keep serving
            self.last_error = f"cannot open writer: {type(e).__name__}: {e}"
            with self._lock:
                self._running = False
            return
        try:
            while True:
                with self._lock:
                    project = self._project
                    self._rerun = False
                try:
                    self.last_reports = writer.ingest_corpus(self._corpus_root, project)
                    self.last_error = None
                except Exception as e:  # noqa: BLE001 - service boundary: report, keep serving
                    self.last_error = f"{type(e).__name__}: {e}"
                self.runs += 1
                with self._lock:
                    if not self._rerun:
                        self._running = False
                        return
        finally:
            writer.close()
