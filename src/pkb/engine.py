"""Concurrent ingest engine using asyncio + to_thread().

Provides:
- EventCollector: bounded async queue with time-window batch drain
- ChunkBuffer: accumulate chunks for batch flush (Phase 2 activation)
- IngestEngine: concurrent file processing with Semaphore-based limits
- IngestResult / IngestStats: result tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pkb.models.config import ConcurrencyConfig

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of processing a single file."""

    path: Path
    status: str  # "ok", "skipped", "merged", "error"
    bundle_id: str | None = None
    error: str | None = None
    platform: str | None = None


@dataclass
class IngestStats:
    """Aggregate statistics for a batch of ingest operations."""

    success: int = 0
    skipped: int = 0
    errors: int = 0
    total: int = 0

    @classmethod
    def from_results(cls, results: list[IngestResult]) -> IngestStats:
        stats = cls(total=len(results))
        for r in results:
            if r.status in ("ok", "merged"):
                stats.success += 1
            elif r.status == "skipped":
                stats.skipped += 1
            elif r.status == "error":
                stats.errors += 1
        return stats


class EventCollector:
    """Bounded async queue with time-window batch drain and path dedup.

    Replaces timer-per-file approach: events go into a bounded asyncio.Queue,
    drain_batch() collects up to max_batch_size items within batch_window seconds.
    """

    def __init__(self, config: ConcurrencyConfig) -> None:
        self._queue: asyncio.Queue[Path] = asyncio.Queue(maxsize=config.max_queue_size)
        self._batch_window = config.batch_window
        self._max_batch_size = config.max_batch_size
        self._dedup_window = 2.0  # seconds to suppress duplicate paths
        self._seen: dict[str, float] = {}

    async def put(self, path: Path) -> None:
        """Add a file path to the queue, deduplicating recent entries."""
        key = str(path)
        now = time.monotonic()

        # Clean expired entries
        expired = [k for k, t in self._seen.items() if now - t > self._dedup_window]
        for k in expired:
            del self._seen[k]

        # Dedup check
        if key in self._seen:
            return

        self._seen[key] = now
        await self._queue.put(path)

    async def drain_batch(self) -> list[Path]:
        """Collect paths for up to batch_window seconds, return at most max_batch_size.

        Returns empty list if no events arrive within the window.
        """
        batch: list[Path] = []

        # Wait for first item with timeout
        try:
            first = await asyncio.wait_for(
                self._queue.get(), timeout=self._batch_window,
            )
            batch.append(first)
        except asyncio.TimeoutError:
            return []

        # Drain remaining items (non-blocking) up to max_batch_size
        deadline = time.monotonic() + self._batch_window
        while len(batch) < self._max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(
                    self._queue.get(), timeout=min(remaining, 0.05),
                )
                batch.append(item)
            except asyncio.TimeoutError:
                break

        return batch


class ChunkBuffer:
    """Accumulate ChromaDB chunks and batch-flush when threshold is reached.

    Phase 1: disabled by default (chunk_buffer_size=0).
    Phase 2: enable with chunk_buffer_size > 0 for batch upsert.
    """

    def __init__(
        self,
        config: ConcurrencyConfig,
        *,
        flush_fn: Callable | None = None,
    ) -> None:
        self._threshold = config.chunk_buffer_size
        self._flush_interval = config.chunk_flush_interval
        self._flush_fn = flush_fn
        self._buffer: list[dict] = []
        self._lock = asyncio.Lock()
        self._last_flush = time.monotonic()

    @property
    def enabled(self) -> bool:
        return self._threshold > 0

    async def add(self, chunks: list[dict]) -> None:
        """Add chunks to buffer, auto-flush if threshold reached."""
        if not self.enabled:
            return
        async with self._lock:
            self._buffer.extend(chunks)
            if len(self._buffer) >= self._threshold:
                await self._do_flush()

    async def flush(self) -> None:
        """Force flush all buffered chunks."""
        async with self._lock:
            if self._buffer:
                await self._do_flush()

    async def maybe_flush(self) -> None:
        """Flush if interval has elapsed since last flush."""
        if not self.enabled:
            return
        if time.monotonic() - self._last_flush >= self._flush_interval:
            await self.flush()

    async def _do_flush(self) -> None:
        """Internal flush (must be called with lock held)."""
        if self._flush_fn and self._buffer:
            chunks = self._buffer[:]
            self._buffer.clear()
            self._last_flush = time.monotonic()
            await self._flush_fn(chunks)


class IngestEngine:
    """Concurrent file ingest engine using asyncio.to_thread().

    Wraps a sync ingest_fn with Semaphore-based concurrency limits.
    Used by both `pkb batch` and `pkb watch` commands.
    """

    def __init__(
        self,
        *,
        ingest_fn: Callable[[Path], dict | None],
        concurrency: ConcurrencyConfig,
        progress_callback: Callable[[IngestResult], None] | None = None,
    ) -> None:
        self._ingest_fn = ingest_fn
        self._config = concurrency
        self._progress = progress_callback
        self._file_semaphore = asyncio.Semaphore(concurrency.max_concurrent_files)

    async def ingest_one(self, path: Path) -> IngestResult:
        """Process a single file with concurrency control."""
        async with self._file_semaphore:
            try:
                result = await asyncio.to_thread(self._ingest_fn, path)
                if result is None:
                    return IngestResult(path=path, status="skipped")
                if result.get("merged"):
                    return IngestResult(
                        path=path, status="merged",
                        bundle_id=result.get("bundle_id"),
                        platform=result.get("platform"),
                    )
                return IngestResult(
                    path=path, status="ok", bundle_id=result.get("bundle_id"),
                )
            except Exception as e:
                logger.exception("Error ingesting %s", path)
                return IngestResult(path=path, status="error", error=str(e))

    async def ingest_batch(self, paths: list[Path]) -> IngestStats:
        """Process multiple files concurrently."""
        tasks = [asyncio.create_task(self.ingest_one(p)) for p in paths]
        results: list[IngestResult] = []

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if self._progress:
                self._progress(result)

        return IngestStats.from_results(results)

    async def run_watch(
        self,
        collector: EventCollector,
        *,
        shutdown_event: asyncio.Event,
    ) -> None:
        """Main watch loop: drain batches from collector and process them."""
        while not shutdown_event.is_set():
            batch = await collector.drain_batch()
            if batch:
                await self.ingest_batch(batch)
            # Check shutdown between iterations
            if shutdown_event.is_set():
                break
