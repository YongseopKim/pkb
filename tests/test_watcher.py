"""Tests for KBWatcher, JSONLEventHandler, and AsyncFileEventHandler."""

import asyncio
import queue
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pkb.constants import DONE_DIR_NAME
from pkb.watcher import JSONLEventHandler, KBWatcher


class TestJSONLEventHandler:
    def test_only_handles_jsonl_files(self, tmp_path):
        q = queue.Queue()
        handler = JSONLEventHandler(file_queue=q, debounce_seconds=0.1)

        event = MagicMock()
        event.src_path = str(tmp_path / "test.jsonl")
        event.is_directory = False
        handler.on_created(event)

        time.sleep(0.3)
        assert not q.empty()
        assert q.get_nowait() == Path(tmp_path / "test.jsonl")

    def test_handles_md_files(self, tmp_path):
        q = queue.Queue()
        handler = JSONLEventHandler(file_queue=q, debounce_seconds=0.1)

        event = MagicMock()
        event.src_path = str(tmp_path / "claude.md")
        event.is_directory = False
        handler.on_created(event)

        time.sleep(0.3)
        assert q.get_nowait() == Path(tmp_path / "claude.md")

    def test_ignores_unsupported_files(self, tmp_path):
        q = queue.Queue()
        handler = JSONLEventHandler(file_queue=q, debounce_seconds=0.1)

        event = MagicMock()
        event.src_path = str(tmp_path / "test.txt")
        event.is_directory = False
        handler.on_created(event)

        time.sleep(0.3)
        assert q.empty()

    def test_ignores_directories(self, tmp_path):
        q = queue.Queue()
        handler = JSONLEventHandler(file_queue=q, debounce_seconds=0.1)

        event = MagicMock()
        event.src_path = str(tmp_path / "somedir")
        event.is_directory = True
        handler.on_created(event)

        time.sleep(0.3)
        assert q.empty()

    def test_debounce_deduplicates(self, tmp_path):
        """Multiple rapid events for same file should only trigger once."""
        q = queue.Queue()
        handler = JSONLEventHandler(file_queue=q, debounce_seconds=0.5)

        event = MagicMock()
        event.src_path = str(tmp_path / "test.jsonl")
        event.is_directory = False

        # Fire 3 events rapidly
        handler.on_created(event)
        handler.on_created(event)
        handler.on_created(event)

        time.sleep(0.8)
        assert q.qsize() == 1

    def test_different_files_handled_separately(self, tmp_path):
        q = queue.Queue()
        handler = JSONLEventHandler(file_queue=q, debounce_seconds=0.1)

        event1 = MagicMock()
        event1.src_path = str(tmp_path / "a.jsonl")
        event1.is_directory = False
        event2 = MagicMock()
        event2.src_path = str(tmp_path / "b.jsonl")
        event2.is_directory = False

        handler.on_created(event1)
        handler.on_created(event2)

        time.sleep(0.3)
        assert q.qsize() == 2

    def test_also_handles_modified_events(self, tmp_path):
        q = queue.Queue()
        handler = JSONLEventHandler(file_queue=q, debounce_seconds=0.1)

        event = MagicMock()
        event.src_path = str(tmp_path / "test.jsonl")
        event.is_directory = False
        handler.on_modified(event)

        time.sleep(0.3)
        assert q.qsize() == 1

    def test_ignores_done_dir_events(self, tmp_path):
        """.done/ 내부 파일 이벤트는 무시."""
        q = queue.Queue()
        handler = JSONLEventHandler(file_queue=q, debounce_seconds=0.1)

        event = MagicMock()
        event.src_path = str(tmp_path / DONE_DIR_NAME / "old.jsonl")
        event.is_directory = False
        handler.on_created(event)

        time.sleep(0.3)
        assert q.empty()

    def test_handles_subdirectory_file_event(self, tmp_path):
        """서브디렉토리 파일 이벤트 정상 처리."""
        q = queue.Queue()
        handler = JSONLEventHandler(file_queue=q, debounce_seconds=0.1)

        event = MagicMock()
        event.src_path = str(tmp_path / "PKB" / "chatgpt.md")
        event.is_directory = False
        handler.on_created(event)

        time.sleep(0.3)
        assert not q.empty()
        assert q.get_nowait() == Path(tmp_path / "PKB" / "chatgpt.md")

    def test_ignores_nested_done_dir_events(self, tmp_path):
        """서브디렉토리 내 .done/ 파일 이벤트도 무시."""
        q = queue.Queue()
        handler = JSONLEventHandler(file_queue=q, debounce_seconds=0.1)

        event = MagicMock()
        event.src_path = str(tmp_path / "PKB" / DONE_DIR_NAME / "old.md")
        event.is_directory = False
        handler.on_created(event)

        time.sleep(0.3)
        assert q.empty()


class TestKBWatcher:
    def test_construction(self, tmp_path):
        callback = MagicMock()
        watcher = KBWatcher(
            watch_dirs=[tmp_path],
            on_new_file=callback,
        )
        assert watcher is not None

    @patch("pkb.watcher.Observer")
    def test_start_schedules_observers(self, mock_observer_cls, tmp_path):
        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()

        callback = MagicMock()
        watcher = KBWatcher(
            watch_dirs=[watch_dir],
            on_new_file=callback,
        )

        mock_observer = MagicMock()
        mock_observer_cls.return_value = mock_observer

        watcher.start()
        mock_observer.schedule.assert_called_once()
        # recursive=True로 서브디렉토리까지 감지
        _, kwargs = mock_observer.schedule.call_args
        assert kwargs.get("recursive") is True
        mock_observer.start.assert_called_once()

        watcher.stop()
        mock_observer.stop.assert_called_once()

    def test_creates_missing_dirs(self, tmp_path):
        watch_dir = tmp_path / "new_inbox"
        callback = MagicMock()
        KBWatcher(
            watch_dirs=[watch_dir],
            on_new_file=callback,
        )
        assert watch_dir.exists()


class TestQueueWorkerPattern:
    """Queue + Worker Thread를 통한 순차 처리 검증."""

    @patch("pkb.watcher.Observer")
    def test_sequential_processing(self, mock_observer_cls, tmp_path):
        """여러 파일이 동시에 들어와도 callback이 순차 실행되어야 함."""
        call_order = []

        def slow_callback(path):
            call_order.append(str(path.name))
            time.sleep(0.05)

        watcher = KBWatcher(
            watch_dirs=[tmp_path],
            on_new_file=slow_callback,
            debounce_seconds=0.01,
        )

        mock_observer = MagicMock()
        mock_observer_cls.return_value = mock_observer

        watcher.start()

        # Queue에 직접 파일 추가 (debounce bypass)
        watcher._queue.put(Path(tmp_path / "a.jsonl"))
        watcher._queue.put(Path(tmp_path / "b.jsonl"))
        watcher._queue.put(Path(tmp_path / "c.jsonl"))

        time.sleep(0.5)
        watcher.stop()

        assert call_order == ["a.jsonl", "b.jsonl", "c.jsonl"]

    @patch("pkb.watcher.Observer")
    def test_graceful_shutdown(self, mock_observer_cls, tmp_path):
        """stop() 호출 시 worker thread가 정상 종료되어야 함."""
        callback = MagicMock()
        watcher = KBWatcher(
            watch_dirs=[tmp_path],
            on_new_file=callback,
        )

        mock_observer = MagicMock()
        mock_observer_cls.return_value = mock_observer

        watcher.start()
        assert watcher._worker_thread.is_alive()

        watcher.stop()
        assert not watcher._worker_thread.is_alive()

    @patch("pkb.watcher.Observer")
    def test_callback_error_does_not_crash_worker(self, mock_observer_cls, tmp_path):
        """callback 에러가 발생해도 worker thread가 계속 동작해야 함."""
        results = []

        def flaky_callback(path):
            if path.name == "bad.jsonl":
                raise ValueError("Test error")
            results.append(path.name)

        watcher = KBWatcher(
            watch_dirs=[tmp_path],
            on_new_file=flaky_callback,
            debounce_seconds=0.01,
        )

        mock_observer = MagicMock()
        mock_observer_cls.return_value = mock_observer

        watcher.start()

        watcher._queue.put(Path(tmp_path / "good1.jsonl"))
        watcher._queue.put(Path(tmp_path / "bad.jsonl"))
        watcher._queue.put(Path(tmp_path / "good2.jsonl"))

        time.sleep(0.5)
        watcher.stop()

        assert results == ["good1.jsonl", "good2.jsonl"]


class TestAsyncFileEventHandler:
    """Tests for watchdog → asyncio bridge handler."""

    @pytest.mark.asyncio
    async def test_jsonl_event_puts_to_collector(self, tmp_path):
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig
        from pkb.watcher import AsyncFileEventHandler

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        loop = asyncio.get_running_loop()
        collector = EventCollector(config)

        handler = AsyncFileEventHandler(loop=loop, collector=collector)

        event = MagicMock()
        event.src_path = str(tmp_path / "test.jsonl")
        event.is_directory = False

        handler.on_created(event)
        await asyncio.sleep(0.05)

        batch = await collector.drain_batch()
        assert len(batch) == 1
        assert batch[0] == Path(tmp_path / "test.jsonl")

    @pytest.mark.asyncio
    async def test_md_event_accepted(self, tmp_path):
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig
        from pkb.watcher import AsyncFileEventHandler

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        loop = asyncio.get_running_loop()
        collector = EventCollector(config)
        handler = AsyncFileEventHandler(loop=loop, collector=collector)

        event = MagicMock()
        event.src_path = str(tmp_path / "claude.md")
        event.is_directory = False

        handler.on_created(event)
        await asyncio.sleep(0.05)

        batch = await collector.drain_batch()
        assert len(batch) == 1

    @pytest.mark.asyncio
    async def test_ignores_unsupported_extensions(self, tmp_path):
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig
        from pkb.watcher import AsyncFileEventHandler

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        loop = asyncio.get_running_loop()
        collector = EventCollector(config)
        handler = AsyncFileEventHandler(loop=loop, collector=collector)

        event = MagicMock()
        event.src_path = str(tmp_path / "test.txt")
        event.is_directory = False

        handler.on_created(event)
        await asyncio.sleep(0.05)

        batch = await collector.drain_batch()
        assert len(batch) == 0

    @pytest.mark.asyncio
    async def test_ignores_directories(self, tmp_path):
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig
        from pkb.watcher import AsyncFileEventHandler

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        loop = asyncio.get_running_loop()
        collector = EventCollector(config)
        handler = AsyncFileEventHandler(loop=loop, collector=collector)

        event = MagicMock()
        event.src_path = str(tmp_path / "somedir")
        event.is_directory = True

        handler.on_created(event)
        await asyncio.sleep(0.05)

        batch = await collector.drain_batch()
        assert len(batch) == 0

    @pytest.mark.asyncio
    async def test_on_modified_also_handled(self, tmp_path):
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig
        from pkb.watcher import AsyncFileEventHandler

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        loop = asyncio.get_running_loop()
        collector = EventCollector(config)
        handler = AsyncFileEventHandler(loop=loop, collector=collector)

        event = MagicMock()
        event.src_path = str(tmp_path / "test.jsonl")
        event.is_directory = False

        handler.on_modified(event)
        await asyncio.sleep(0.05)

        batch = await collector.drain_batch()
        assert len(batch) == 1

    @pytest.mark.asyncio
    async def test_ignores_done_dir_events(self, tmp_path):
        """.done/ 내부 파일 이벤트는 무시."""
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig
        from pkb.watcher import AsyncFileEventHandler

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        loop = asyncio.get_running_loop()
        collector = EventCollector(config)
        handler = AsyncFileEventHandler(loop=loop, collector=collector)

        event = MagicMock()
        event.src_path = str(tmp_path / DONE_DIR_NAME / "old.jsonl")
        event.is_directory = False

        handler.on_created(event)
        await asyncio.sleep(0.05)

        batch = await collector.drain_batch()
        assert len(batch) == 0

    @pytest.mark.asyncio
    async def test_handles_subdirectory_file_event(self, tmp_path):
        """서브디렉토리 파일 이벤트 정상 처리."""
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig
        from pkb.watcher import AsyncFileEventHandler

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        loop = asyncio.get_running_loop()
        collector = EventCollector(config)
        handler = AsyncFileEventHandler(loop=loop, collector=collector)

        event = MagicMock()
        event.src_path = str(tmp_path / "PKB" / "chatgpt.md")
        event.is_directory = False

        handler.on_created(event)
        await asyncio.sleep(0.05)

        batch = await collector.drain_batch()
        assert len(batch) == 1
        assert batch[0] == Path(tmp_path / "PKB" / "chatgpt.md")

    @pytest.mark.asyncio
    async def test_ignores_nested_done_dir_events(self, tmp_path):
        """서브디렉토리 내 .done/ 파일 이벤트도 무시."""
        from pkb.engine import EventCollector
        from pkb.models.config import ConcurrencyConfig
        from pkb.watcher import AsyncFileEventHandler

        config = ConcurrencyConfig(max_queue_size=100, batch_window=0.1, max_batch_size=10)
        loop = asyncio.get_running_loop()
        collector = EventCollector(config)
        handler = AsyncFileEventHandler(loop=loop, collector=collector)

        event = MagicMock()
        event.src_path = str(tmp_path / "PKB" / DONE_DIR_NAME / "old.md")
        event.is_directory = False

        handler.on_created(event)
        await asyncio.sleep(0.05)

        batch = await collector.drain_batch()
        assert len(batch) == 0
