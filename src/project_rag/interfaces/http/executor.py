"""One store thread with a bounded queue and per-call deadlines (spec §6.3, A8).

SQLite connections are thread-affine, so every read-path call (search, open_source, status,
feedback) runs on ONE dedicated thread, serialised. At most `max_waiting` calls may wait behind
the running one; the next caller gets `Overloaded` (the HTTP layer answers 503 + Retry-After).
A call that exceeds its deadline returns `DeadlineExceeded` to the caller; the work itself
finishes on the store thread (Python threads cannot be cancelled) and its result is dropped.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")


class Overloaded(RuntimeError):
    """The bounded queue is full."""


class DeadlineExceeded(TimeoutError):
    """The call did not finish within its deadline."""


class StoreExecutor:
    def __init__(self, max_waiting: int = 16, thread_name: str = "assist-store") -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)
        self._max_waiting = max_waiting
        self._pending = 0  # running + waiting
        self._lock = threading.Lock()
        self.rejected = 0

    @property
    def pending(self) -> int:
        return self._pending

    @property
    def saturated(self) -> bool:
        """True when one more call would be refused (running + max_waiting already queued)."""
        return self._pending >= self._max_waiting + 1

    def submit_sync(self, fn: Callable[[], T]) -> T:
        """Run on the store thread and block (startup / admin use; no queue bound)."""
        return self._pool.submit(fn).result()

    async def run(self, fn: Callable[[], T], deadline_s: float) -> T:
        with self._lock:
            if self._pending >= self._max_waiting + 1:
                self.rejected += 1
                raise Overloaded(f"{self._pending} requests in flight")
            self._pending += 1
        future = self._pool.submit(fn)

        def _done(_f: object) -> None:
            with self._lock:
                self._pending -= 1

        future.add_done_callback(_done)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout=deadline_s)
        except TimeoutError as e:
            raise DeadlineExceeded(f"deadline {deadline_s:.0f}s exceeded") from e

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
