"""Watch KB directories for new input files and auto-ingest."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from pkb.constants import DONE_DIR_NAME
from pkb.parser.directory import SUPPORTED_EXTENSIONS

if TYPE_CHECKING:
    from pkb.engine import EventCollector

logger = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_SECONDS = 5.0


class JSONLEventHandler(FileSystemEventHandler):
    """Handles input file (.jsonl, .md) creation/modification events with debounce.

    Instead of calling a callback directly, enqueues file paths for sequential
    processing by a worker thread.
    """

    def __init__(
        self,
        file_queue: queue.Queue,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
    ) -> None:
        super().__init__()
        self._queue = file_queue
        self._debounce = debounce_seconds
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _handle_event(self, event) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix not in SUPPORTED_EXTENSIONS:
            return
        if DONE_DIR_NAME in path.parts:
            return

        with self._lock:
            key = str(path)
            # Cancel existing timer for this file
            if key in self._timers:
                self._timers[key].cancel()

            # Start new debounced timer
            timer = threading.Timer(
                self._debounce, self._fire_callback, args=[path]
            )
            self._timers[key] = timer
            timer.start()

    def _fire_callback(self, path: Path) -> None:
        with self._lock:
            self._timers.pop(str(path), None)
        self._queue.put(path)

    def on_created(self, event) -> None:
        self._handle_event(event)

    def on_modified(self, event) -> None:
        self._handle_event(event)


class KBWatcher:
    """Watches one or more KB directories for new input files.

    Uses a Queue + single Worker Thread pattern to ensure sequential processing,
    preventing race conditions from concurrent ingest operations.
    """

    def __init__(
        self,
        *,
        watch_dirs: list[Path],
        on_new_file: Callable[[Path], None],
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
    ) -> None:
        self._watch_dirs = watch_dirs
        self._callback = on_new_file
        self._queue: queue.Queue[Path | None] = queue.Queue()
        self._handler = JSONLEventHandler(
            file_queue=self._queue,
            debounce_seconds=debounce_seconds,
        )
        self._observer: Observer | None = None
        self._worker_thread: threading.Thread | None = None

        # Ensure watch dirs exist
        for d in self._watch_dirs:
            d.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start watching directories and worker thread."""
        self._observer = Observer()
        for d in self._watch_dirs:
            self._observer.schedule(self._handler, str(d), recursive=True)
            logger.info("Watching %s", d)
        self._observer.start()

        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True, name="pkb-watcher-worker",
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop watching directories and worker thread."""
        # Send sentinel to stop worker
        self._queue.put(None)
        if self._worker_thread:
            self._worker_thread.join(timeout=10)

        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

    def _worker(self) -> None:
        """Worker thread: sequentially process files from the queue."""
        while True:
            path = self._queue.get()
            if path is None:
                break
            try:
                self._callback(path)
            except Exception:
                logger.exception("Error processing %s", path)


class AsyncFileEventHandler(FileSystemEventHandler):
    """Bridges watchdog filesystem events to an asyncio EventCollector.

    Unlike JSONLEventHandler (timer-per-file + threading.Queue), this handler
    uses asyncio.run_coroutine_threadsafe() to enqueue paths into EventCollector,
    which handles dedup and batch drain internally.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        collector: EventCollector,
    ) -> None:
        super().__init__()
        self._loop = loop
        self._collector = collector

    def _handle_event(self, event) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix not in SUPPORTED_EXTENSIONS:
            return
        if DONE_DIR_NAME in path.parts:
            return

        asyncio.run_coroutine_threadsafe(self._collector.put(path), self._loop)

    def on_created(self, event) -> None:
        self._handle_event(event)

    def on_modified(self, event) -> None:
        self._handle_event(event)
