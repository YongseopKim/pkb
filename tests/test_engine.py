"""Tests for IngestEngine, EventCollector, and ChunkBuffer."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pkb.models.config import ConcurrencyConfig

# -- EventCollector Tests --


class TestEventCollector:
    """Tests for bounded async event queue with dedup and time-window drain."""

    @pytest.fixture
    def config(self):
        return ConcurrencyConfig(
            max_queue_size=100,
            batch_window=0.2,
            max_batch_size=5,
        )

    @pytest.mark.asyncio
    async def test_put_and_drain(self, config):
        from pkb.engine import EventCollector

        collector = EventCollector(config)
        await collector.put(Path("/tmp/a.jsonl"))
        await collector.put(Path("/tmp/b.jsonl"))

        batch = await collector.drain_batch()
        assert len(batch) == 2
        assert Path("/tmp/a.jsonl") in batch
        assert Path("/tmp/b.jsonl") in batch

    @pytest.mark.asyncio
    async def test_dedup_within_window(self, config):
        """Same path within 2 seconds should be deduped."""
        from pkb.engine import EventCollector

        collector = EventCollector(config)
        await collector.put(Path("/tmp/a.jsonl"))
        await collector.put(Path("/tmp/a.jsonl"))  # duplicate

        batch = await collector.drain_batch()
        assert len(batch) == 1

    @pytest.mark.asyncio
    async def test_drain_respects_max_batch_size(self, config):
        from pkb.engine import EventCollector

        collector = EventCollector(config)
        for i in range(10):
            await collector.put(Path(f"/tmp/file_{i}.jsonl"))

        batch = await collector.drain_batch()
        assert len(batch) == 5  # max_batch_size

    @pytest.mark.asyncio
    async def test_drain_empty_returns_empty(self, config):
        from pkb.engine import EventCollector

        collector = EventCollector(config)
        batch = await collector.drain_batch()
        assert batch == []

    @pytest.mark.asyncio
    async def test_dedup_expires_after_window(self, config):
        """Same path after dedup window should NOT be deduped."""
        from pkb.engine import EventCollector

        collector = EventCollector(
            ConcurrencyConfig(
                max_queue_size=100,
                batch_window=0.1,
                max_batch_size=50,
            )
        )
        collector._dedup_window = 0.1  # 100ms dedup window for test
        await collector.put(Path("/tmp/a.jsonl"))
        batch1 = await collector.drain_batch()
        assert len(batch1) == 1

        await asyncio.sleep(0.15)  # Wait for dedup to expire

        await collector.put(Path("/tmp/a.jsonl"))
        batch2 = await collector.drain_batch()
        assert len(batch2) == 1


# -- ChunkBuffer Tests --


class TestChunkBuffer:
    """Tests for chunk accumulation and batch flush."""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        """ChunkBuffer with threshold=0 should be disabled."""
        from pkb.engine import ChunkBuffer

        config = ConcurrencyConfig(chunk_buffer_size=0)
        mock_flush_fn = AsyncMock()
        buffer = ChunkBuffer(config, flush_fn=mock_flush_fn)
        assert not buffer.enabled

    @pytest.mark.asyncio
    async def test_enabled_with_threshold(self):
        from pkb.engine import ChunkBuffer

        config = ConcurrencyConfig(chunk_buffer_size=100)
        mock_flush_fn = AsyncMock()
        buffer = ChunkBuffer(config, flush_fn=mock_flush_fn)
        assert buffer.enabled

    @pytest.mark.asyncio
    async def test_add_below_threshold(self):
        from pkb.engine import ChunkBuffer

        config = ConcurrencyConfig(chunk_buffer_size=100)
        mock_flush_fn = AsyncMock()
        buffer = ChunkBuffer(config, flush_fn=mock_flush_fn)

        await buffer.add([{"id": "1", "doc": "test"}])
        assert len(buffer._buffer) == 1
        mock_flush_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_flush_on_threshold(self):
        from pkb.engine import ChunkBuffer

        config = ConcurrencyConfig(chunk_buffer_size=2)
        mock_flush_fn = AsyncMock()
        buffer = ChunkBuffer(config, flush_fn=mock_flush_fn)

        await buffer.add([{"id": "1"}, {"id": "2"}])
        mock_flush_fn.assert_called_once()
        assert len(buffer._buffer) == 0

    @pytest.mark.asyncio
    async def test_explicit_flush(self):
        from pkb.engine import ChunkBuffer

        config = ConcurrencyConfig(chunk_buffer_size=100)
        mock_flush_fn = AsyncMock()
        buffer = ChunkBuffer(config, flush_fn=mock_flush_fn)

        await buffer.add([{"id": "1"}])
        await buffer.flush()
        mock_flush_fn.assert_called_once()
        assert len(buffer._buffer) == 0


# -- IngestResult / IngestStats --


class TestIngestModels:
    def test_ingest_result_success(self):
        from pkb.engine import IngestResult

        r = IngestResult(path=Path("/tmp/a.jsonl"), status="ok", bundle_id="test-id")
        assert r.status == "ok"
        assert r.error is None

    def test_ingest_result_error(self):
        from pkb.engine import IngestResult

        r = IngestResult(path=Path("/tmp/a.jsonl"), status="error", error="fail")
        assert r.status == "error"
        assert r.error == "fail"

    def test_ingest_stats_from_results(self):
        from pkb.engine import IngestResult, IngestStats

        results = [
            IngestResult(path=Path("/tmp/a.jsonl"), status="ok", bundle_id="a"),
            IngestResult(path=Path("/tmp/b.jsonl"), status="skipped"),
            IngestResult(path=Path("/tmp/c.jsonl"), status="error", error="oops"),
        ]
        stats = IngestStats.from_results(results)
        assert stats.success == 1
        assert stats.skipped == 1
        assert stats.errors == 1
        assert stats.total == 3


# -- IngestEngine Tests --


class TestIngestEngine:
    """Tests for async concurrent ingest engine."""

    @pytest.fixture
    def config(self):
        return ConcurrencyConfig(
            max_concurrent_files=2,
            max_concurrent_llm=2,
        )

    @pytest.mark.asyncio
    async def test_ingest_one_success(self, config):
        from pkb.engine import IngestEngine

        def fake_ingest(path):
            return {"bundle_id": f"id-{path.stem}"}

        engine = IngestEngine(ingest_fn=fake_ingest, concurrency=config)
        result = await engine.ingest_one(Path("/tmp/test.jsonl"))
        assert result.status == "ok"
        assert result.bundle_id == "id-test"

    @pytest.mark.asyncio
    async def test_ingest_one_skip(self, config):
        from pkb.engine import IngestEngine

        def fake_ingest(path):
            return None  # duplicate

        engine = IngestEngine(ingest_fn=fake_ingest, concurrency=config)
        result = await engine.ingest_one(Path("/tmp/test.jsonl"))
        assert result.status == "skipped"

    @pytest.mark.asyncio
    async def test_ingest_one_error(self, config):
        from pkb.engine import IngestEngine

        def fake_ingest(path):
            raise RuntimeError("LLM failed")

        engine = IngestEngine(ingest_fn=fake_ingest, concurrency=config)
        result = await engine.ingest_one(Path("/tmp/test.jsonl"))
        assert result.status == "error"
        assert "LLM failed" in result.error

    @pytest.mark.asyncio
    async def test_ingest_batch(self, config):
        from pkb.engine import IngestEngine

        def fake_ingest(path):
            return {"bundle_id": f"id-{path.stem}"}

        engine = IngestEngine(ingest_fn=fake_ingest, concurrency=config)
        paths = [Path(f"/tmp/file_{i}.jsonl") for i in range(5)]
        stats = await engine.ingest_batch(paths)
        assert stats.success == 5
        assert stats.total == 5

    @pytest.mark.asyncio
    async def test_ingest_batch_mixed(self, config):
        from pkb.engine import IngestEngine

        call_count = 0

        def fake_ingest(path):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise RuntimeError("fail")
            if call_count % 2 == 0:
                return None
            return {"bundle_id": f"id-{call_count}"}

        engine = IngestEngine(ingest_fn=fake_ingest, concurrency=config)
        paths = [Path(f"/tmp/file_{i}.jsonl") for i in range(6)]
        stats = await engine.ingest_batch(paths)
        assert stats.total == 6
        assert stats.success + stats.skipped + stats.errors == 6

    @pytest.mark.asyncio
    async def test_concurrency_limit(self, config):
        """Verify semaphore limits concurrent execution."""
        from pkb.engine import IngestEngine

        config = ConcurrencyConfig(max_concurrent_files=2)
        max_concurrent = 0
        current_concurrent = 0

        def slow_ingest(path):
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            if current_concurrent > max_concurrent:
                max_concurrent = current_concurrent
            time.sleep(0.05)
            current_concurrent -= 1
            return {"bundle_id": "x"}

        engine = IngestEngine(ingest_fn=slow_ingest, concurrency=config)
        paths = [Path(f"/tmp/file_{i}.jsonl") for i in range(6)]
        await engine.ingest_batch(paths)

        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_progress_callback(self, config):
        from pkb.engine import IngestEngine

        results_seen = []

        def on_progress(result):
            results_seen.append(result)

        def fake_ingest(path):
            return {"bundle_id": "test"}

        engine = IngestEngine(
            ingest_fn=fake_ingest, concurrency=config, progress_callback=on_progress,
        )
        await engine.ingest_batch([Path("/tmp/a.jsonl"), Path("/tmp/b.jsonl")])
        assert len(results_seen) == 2

    @pytest.mark.asyncio
    async def test_run_watch_processes_batches(self, config):
        from pkb.engine import EventCollector, IngestEngine

        config = ConcurrencyConfig(
            max_concurrent_files=2,
            batch_window=0.1,
            max_batch_size=10,
        )

        results = []

        def fake_ingest(path):
            results.append(path)
            return {"bundle_id": f"id-{path.stem}"}

        engine = IngestEngine(ingest_fn=fake_ingest, concurrency=config)
        collector = EventCollector(config)

        # Add files before starting
        await collector.put(Path("/tmp/a.jsonl"))
        await collector.put(Path("/tmp/b.jsonl"))

        shutdown = asyncio.Event()

        async def stop_after_delay():
            await asyncio.sleep(0.3)
            shutdown.set()

        asyncio.create_task(stop_after_delay())
        await engine.run_watch(collector, shutdown_event=shutdown)

        assert len(results) == 2
