"""Tests for Reindexer class."""

from unittest.mock import MagicMock

import pytest
import yaml

from pkb.models.config import EmbeddingConfig
from pkb.reindex import Reindexer


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.list_all_bundle_ids.return_value = []
    return repo


@pytest.fixture
def mock_chunk_store():
    return MagicMock()


@pytest.fixture
def kb_dir(tmp_path):
    """Create a KB directory with a sample bundle."""
    bundle_dir = tmp_path / "bundles" / "20260101-test-abc1"
    bundle_dir.mkdir(parents=True)

    # Write _bundle.md with frontmatter
    bundle_md = {
        "id": "20260101-test-abc1",
        "question": "테스트 질문",
        "summary": "테스트 요약",
        "slug": "test",
        "domains": ["dev"],
        "topics": ["python"],
        "pending_topics": [],
        "platforms": ["claude"],
        "created_at": "2026-01-01T00:00:00+00:00",
        "consensus": None,
        "divergence": None,
    }
    content = f"---\n{yaml.dump(bundle_md, allow_unicode=True, default_flow_style=False)}---\n"
    (bundle_dir / "_bundle.md").write_text(content, encoding="utf-8")

    # Write platform MD
    platform_md = "---\nplatform: claude\nsummary: 응답 요약\n---\n\n# Content\nHello world.\n"
    (bundle_dir / "claude.md").write_text(platform_md, encoding="utf-8")

    return tmp_path


@pytest.fixture
def reindexer(mock_repo, mock_chunk_store, kb_dir):
    return Reindexer(
        repo=mock_repo,
        chunk_store=mock_chunk_store,
        kb_path=kb_dir,
        kb_name="personal",
        embedding_config=EmbeddingConfig(),
    )


class TestReindexBundle:
    def test_reindex_single_bundle(self, reindexer, mock_repo, mock_chunk_store):
        result = reindexer.reindex_bundle("20260101-test-abc1")
        assert result["bundle_id"] == "20260101-test-abc1"
        assert result["status"] == "updated"
        mock_repo.update_bundle_meta.assert_called_once()
        mock_chunk_store.delete_by_bundle.assert_called_once_with("20260101-test-abc1")
        mock_chunk_store.upsert_chunks.assert_called_once()

    def test_reindex_updates_meta_correctly(self, reindexer, mock_repo):
        reindexer.reindex_bundle("20260101-test-abc1")
        call_kwargs = mock_repo.update_bundle_meta.call_args[1]
        assert call_kwargs["bundle_id"] == "20260101-test-abc1"
        assert call_kwargs["summary"] == "테스트 요약"
        assert call_kwargs["domains"] == ["dev"]
        assert call_kwargs["topics"] == ["python"]

    def test_reindex_missing_bundle_dir(self, reindexer):
        result = reindexer.reindex_bundle("nonexistent-bundle")
        assert result["status"] == "skipped"
        assert "not found" in result["reason"].lower()

    def test_reindex_missing_bundle_md(self, reindexer, kb_dir):
        bundle_dir = kb_dir / "bundles" / "20260102-no-md-abc1"
        bundle_dir.mkdir(parents=True)
        result = reindexer.reindex_bundle("20260102-no-md-abc1")
        assert result["status"] == "skipped"

    def test_reindex_invalid_frontmatter(self, reindexer, kb_dir):
        bundle_dir = kb_dir / "bundles" / "20260103-bad-abc1"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "_bundle.md").write_text(
            "---\nid: bad\n", encoding="utf-8"
        )
        result = reindexer.reindex_bundle("20260103-bad-abc1")
        assert result["status"] == "error"

    def test_reindex_rechunks_platform_md(self, reindexer, mock_chunk_store):
        reindexer.reindex_bundle("20260101-test-abc1")
        chunks = mock_chunk_store.upsert_chunks.call_args[0][0]
        assert len(chunks) > 0
        assert chunks[0]["metadata"]["bundle_id"] == "20260101-test-abc1"


class TestReindexFull:
    def test_full_reindex_scans_all_bundles(self, reindexer, mock_repo):
        result = reindexer.reindex_full()
        assert result["updated"] >= 1
        assert "total" in result

    def test_full_reindex_deletes_orphan_db_records(
        self, reindexer, mock_repo, kb_dir
    ):
        """DB has bundle not on disk → should be deleted."""
        mock_repo.list_all_bundle_ids.return_value = [
            "20260101-test-abc1",  # exists on disk
            "20260199-orphan-9999",  # does NOT exist on disk
        ]
        result = reindexer.reindex_full()
        mock_repo.delete_bundle.assert_called_once_with("20260199-orphan-9999")
        assert result["deleted"] == 1

    def test_full_reindex_with_callback(self, reindexer, mock_repo):
        callback = MagicMock()
        reindexer.reindex_full(progress_callback=callback)
        assert callback.call_count >= 1

    def test_full_reindex_no_orphans(self, reindexer, mock_repo):
        mock_repo.list_all_bundle_ids.return_value = ["20260101-test-abc1"]
        result = reindexer.reindex_full()
        mock_repo.delete_bundle.assert_not_called()
        assert result["deleted"] == 0
