"""Tests for batch processor."""

import json
from unittest.mock import MagicMock

import pytest
import yaml

from pkb.batch import BatchProcessor
from pkb.constants import DONE_DIR_NAME


@pytest.fixture
def sample_source_dir(tmp_path):
    """Create a directory with multiple subdirs each containing JSONL files."""
    source = tmp_path / "exports"
    for name in ["topic1", "topic2", "topic3"]:
        d = source / name
        d.mkdir(parents=True)
        for platform in ["claude", "chatgpt"]:
            lines = [
                json.dumps({
                    "_meta": True,
                    "platform": platform,
                    "url": f"https://example.com/{name}/{platform}",
                    "exported_at": "2026-02-21T06:00:00.000Z",
                    "title": f"{name} - {platform}",
                }),
                json.dumps({
                    "role": "user",
                    "content": f"질문: {name}",
                    "timestamp": "2026-02-21T06:00:01.000Z",
                }),
                json.dumps({
                    "role": "assistant",
                    "content": f"답변: {name}",
                    "timestamp": "2026-02-21T06:00:02.000Z",
                }),
            ]
            (d / f"{platform}.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return source


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.ingest_file.return_value = {
        "bundle_id": "20260221-test-a3f2",
        "platform": "claude",
        "question": "질문",
        "summary": "요약",
        "domains": ["dev"],
        "topics": ["python"],
    }
    return pipeline


class TestBatchProcessor:
    def test_discovers_jsonl_files(self, sample_source_dir, mock_pipeline, tmp_path):
        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=tmp_path / "checkpoint.yaml",
        )
        files = processor.discover_files(sample_source_dir)
        assert len(files) == 6  # 3 dirs × 2 files

    def test_process_all(self, sample_source_dir, mock_pipeline, tmp_path):
        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=tmp_path / "checkpoint.yaml",
        )
        stats = processor.process(sample_source_dir)
        assert stats["success"] == 6
        assert stats["errors"] == 0

    def test_checkpoint_saves(self, sample_source_dir, mock_pipeline, tmp_path):
        checkpoint_path = tmp_path / "checkpoint.yaml"
        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=checkpoint_path,
        )
        processor.process(sample_source_dir)
        assert checkpoint_path.exists()
        data = yaml.safe_load(checkpoint_path.read_text())
        assert len(data["completed"]) == 6

    def test_resume_skips_completed(self, sample_source_dir, mock_pipeline, tmp_path):
        checkpoint_path = tmp_path / "checkpoint.yaml"
        # Pre-populate checkpoint with 2 completed files
        files = sorted(sample_source_dir.rglob("*.jsonl"))
        checkpoint_data = {
            "completed": [str(files[0]), str(files[1])],
            "failed": [],
        }
        checkpoint_path.write_text(yaml.dump(checkpoint_data))

        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=checkpoint_path,
        )
        stats = processor.process(sample_source_dir)
        assert stats["success"] == 4  # 6 - 2 already completed
        assert stats["skipped"] == 2

    def test_max_limit(self, sample_source_dir, mock_pipeline, tmp_path):
        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=tmp_path / "checkpoint.yaml",
            max_files=3,
        )
        stats = processor.process(sample_source_dir)
        assert stats["success"] == 3

    def test_no_resume_ignores_checkpoint(self, sample_source_dir, mock_pipeline, tmp_path):
        checkpoint_path = tmp_path / "checkpoint.yaml"
        files = sorted(sample_source_dir.rglob("*.jsonl"))
        checkpoint_data = {
            "completed": [str(files[0])],
            "failed": [],
        }
        checkpoint_path.write_text(yaml.dump(checkpoint_data))

        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=checkpoint_path,
            resume=False,
        )
        stats = processor.process(sample_source_dir)
        assert stats["success"] == 6  # Processes all, ignoring checkpoint

    def test_handles_pipeline_errors(self, sample_source_dir, tmp_path):
        pipeline = MagicMock()
        pipeline.ingest_file.side_effect = RuntimeError("API error")
        processor = BatchProcessor(
            pipeline=pipeline,
            checkpoint_path=tmp_path / "checkpoint.yaml",
        )
        stats = processor.process(sample_source_dir)
        assert stats["errors"] == 6
        assert stats["success"] == 0

    def test_discover_files_excludes_done_dir(self, tmp_path, mock_pipeline):
        """discover_files()는 .done/ 디렉토리 파일을 제외."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        (inbox / "new.jsonl").write_text("{}")
        done = inbox / DONE_DIR_NAME
        done.mkdir()
        (done / "old.jsonl").write_text("{}")

        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=tmp_path / "checkpoint.yaml",
        )
        files = processor.discover_files(inbox)
        names = [f.name for f in files]
        assert "new.jsonl" in names
        assert "old.jsonl" not in names

    def test_sequential_moves_to_done(self, tmp_path, mock_pipeline):
        """sequential 모드에서 성공 시 watch_dir 기반으로 move_to_done 호출."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        jsonl = inbox / "test.jsonl"
        jsonl.write_text("{}")

        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=tmp_path / "checkpoint.yaml",
            watch_dir=inbox,
        )
        processor.process(inbox)

        # 파일이 .done/으로 이동되었어야 함
        assert not jsonl.exists()
        assert (inbox / DONE_DIR_NAME / "test.jsonl").exists()

    def test_sequential_no_move_without_watch_dir(self, tmp_path, mock_pipeline):
        """watch_dir가 None이면 move_to_done을 호출하지 않음."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        jsonl = inbox / "test.jsonl"
        jsonl.write_text("{}")

        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=tmp_path / "checkpoint.yaml",
        )
        processor.process(inbox)

        # 파일이 원래 위치에 남아있어야 함
        assert jsonl.exists()
        assert not (inbox / DONE_DIR_NAME).exists()


class TestBatchProcessorConcurrent:
    """Tests for concurrent batch processing via IngestEngine."""

    def test_process_with_engine(self, sample_source_dir, mock_pipeline, tmp_path):
        """When engine is provided, process() should use concurrent path."""
        from pkb.engine import IngestEngine
        from pkb.models.config import ConcurrencyConfig

        config = ConcurrencyConfig(max_concurrent_files=2)
        engine = IngestEngine(
            ingest_fn=mock_pipeline.ingest_file, concurrency=config,
        )
        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=tmp_path / "checkpoint.yaml",
            engine=engine,
        )
        stats = processor.process(sample_source_dir)
        assert stats["success"] == 6
        assert stats["errors"] == 0

    def test_concurrent_respects_checkpoint(self, sample_source_dir, mock_pipeline, tmp_path):
        """Concurrent path should skip already-completed files."""
        from pkb.engine import IngestEngine
        from pkb.models.config import ConcurrencyConfig

        config = ConcurrencyConfig(max_concurrent_files=2)
        engine = IngestEngine(
            ingest_fn=mock_pipeline.ingest_file, concurrency=config,
        )
        checkpoint_path = tmp_path / "checkpoint.yaml"
        files = sorted(sample_source_dir.rglob("*.jsonl"))
        checkpoint_data = {
            "completed": [str(files[0]), str(files[1])],
            "failed": [],
        }
        checkpoint_path.write_text(yaml.dump(checkpoint_data))

        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=checkpoint_path,
            engine=engine,
        )
        stats = processor.process(sample_source_dir)
        assert stats["success"] == 4
        assert stats["skipped"] == 2

    def test_concurrent_handles_errors(self, sample_source_dir, tmp_path):
        """Concurrent path should handle individual file errors gracefully."""
        from pkb.engine import IngestEngine
        from pkb.models.config import ConcurrencyConfig

        def flaky_ingest(path):
            if "chatgpt" in str(path):
                raise RuntimeError("fail")
            return {"bundle_id": "ok"}

        config = ConcurrencyConfig(max_concurrent_files=2)
        engine = IngestEngine(ingest_fn=flaky_ingest, concurrency=config)
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file = flaky_ingest

        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=tmp_path / "checkpoint.yaml",
            engine=engine,
        )
        stats = processor.process(sample_source_dir)
        assert stats["success"] == 3  # claude files only
        assert stats["errors"] == 3  # chatgpt files

    def test_concurrent_max_files(self, sample_source_dir, mock_pipeline, tmp_path):
        """Concurrent path should respect max_files limit."""
        from pkb.engine import IngestEngine
        from pkb.models.config import ConcurrencyConfig

        config = ConcurrencyConfig(max_concurrent_files=2)
        engine = IngestEngine(
            ingest_fn=mock_pipeline.ingest_file, concurrency=config,
        )
        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=tmp_path / "checkpoint.yaml",
            max_files=3,
            engine=engine,
        )
        stats = processor.process(sample_source_dir)
        assert stats["success"] == 3

    def test_concurrent_saves_checkpoint(self, sample_source_dir, mock_pipeline, tmp_path):
        """Concurrent path should save checkpoint after processing."""
        from pkb.engine import IngestEngine
        from pkb.models.config import ConcurrencyConfig

        config = ConcurrencyConfig(max_concurrent_files=2)
        engine = IngestEngine(
            ingest_fn=mock_pipeline.ingest_file, concurrency=config,
        )
        checkpoint_path = tmp_path / "checkpoint.yaml"
        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=checkpoint_path,
            engine=engine,
        )
        processor.process(sample_source_dir)
        assert checkpoint_path.exists()
        data = yaml.safe_load(checkpoint_path.read_text())
        assert len(data["completed"]) == 6

    def test_concurrent_checkpoint_excludes_failed(self, sample_source_dir, tmp_path):
        """Concurrent path: 실패한 파일은 checkpoint completed에 포함되지 않아야 함."""
        from pkb.engine import IngestEngine
        from pkb.models.config import ConcurrencyConfig

        def flaky_ingest(path):
            if "chatgpt" in str(path):
                raise RuntimeError("LLM timeout")
            return {"bundle_id": "ok-bundle"}

        config = ConcurrencyConfig(max_concurrent_files=2)
        engine = IngestEngine(ingest_fn=flaky_ingest, concurrency=config)
        mock_pl = MagicMock()
        mock_pl.ingest_file = flaky_ingest

        checkpoint_path = tmp_path / "checkpoint.yaml"
        processor = BatchProcessor(
            pipeline=mock_pl,
            checkpoint_path=checkpoint_path,
            engine=engine,
        )
        stats = processor.process(sample_source_dir)

        assert stats["errors"] == 3  # chatgpt files
        assert stats["success"] == 3  # claude files

        data = yaml.safe_load(checkpoint_path.read_text())
        # Failed files should NOT be in completed list
        assert len(data["completed"]) == 3
        for path_str in data["completed"]:
            assert "chatgpt" not in path_str

    def test_concurrent_move_to_done_only_on_success(self, tmp_path):
        """Concurrent batch: skip/error 결과일 때 move_to_done 호출하지 않음."""
        from pkb.engine import IngestEngine
        from pkb.models.config import ConcurrencyConfig

        # Create watch_dir with files
        watch_dir = tmp_path / "inbox"
        topic_dir = watch_dir / "test_topic"
        topic_dir.mkdir(parents=True)

        for name in ["success.md", "skip.md", "error.md"]:
            (topic_dir / name).write_text(f"# {name}\n\ncontent here for test")

        def mock_ingest(path):
            name = path.name
            if "skip" in name:
                return {"status": "skip_parse_error", "reason": "test skip"}
            if "error" in name:
                raise RuntimeError("test error")
            return {"bundle_id": "ok-bundle"}

        config = ConcurrencyConfig(max_concurrent_files=2)
        engine = IngestEngine(ingest_fn=mock_ingest, concurrency=config)
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file = mock_ingest

        processor = BatchProcessor(
            pipeline=mock_pipeline,
            checkpoint_path=tmp_path / "checkpoint.yaml",
            engine=engine,
            watch_dir=watch_dir,
        )
        processor.process(watch_dir)

        # skip.md should remain in original location (not moved)
        assert (topic_dir / "skip.md").exists()
        # error.md should remain in original location (not moved)
        assert (topic_dir / "error.md").exists()
