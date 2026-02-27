"""Tests for ReembedEngine."""

from unittest.mock import MagicMock

import pytest

from pkb.reembed import ReembedEngine


@pytest.fixture
def tmp_kb(tmp_path):
    """Create a minimal KB directory structure."""
    bundles = tmp_path / "bundles"
    bundles.mkdir()
    return tmp_path


@pytest.fixture
def mock_chunk_store():
    return MagicMock()


@pytest.fixture
def mock_repo():
    return MagicMock()


def _create_bundle(kb_path, bundle_id, md_content="# Test\n\nSome content here."):
    """Helper to create a bundle directory with an MD file."""
    bundle_dir = kb_path / "bundles" / bundle_id
    bundle_dir.mkdir(parents=True)
    md_file = bundle_dir / "claude.md"
    md_file.write_text(md_content, encoding="utf-8")
    return bundle_dir


class TestReembedBundle:
    def test_reembed_single_bundle(self, tmp_kb, mock_chunk_store, mock_repo):
        bundle_id = "20260221-test-a3f2"
        _create_bundle(tmp_kb, bundle_id, "# Test\n\nContent for embedding.")

        mock_repo.find_by_id.return_value = {
            "bundle_id": bundle_id,
            "kb": "personal",
            "platform": "claude",
            "domains": ["dev"],
            "topics": ["python"],
        }

        engine = ReembedEngine(
            kb_path=tmp_kb,
            kb_name="personal",
            chunk_store=mock_chunk_store,
            repo=mock_repo,
            chunk_size=512,
            chunk_overlap=50,
        )
        result = engine.reembed_bundle(bundle_id)
        assert result["status"] == "reembedded"
        assert result["bundle_id"] == bundle_id
        mock_chunk_store.delete_by_bundle.assert_called_once_with(bundle_id)
        mock_chunk_store.upsert_chunks.assert_called_once()

    def test_reembed_missing_bundle_dir(self, tmp_kb, mock_chunk_store, mock_repo):
        engine = ReembedEngine(
            kb_path=tmp_kb,
            kb_name="personal",
            chunk_store=mock_chunk_store,
            repo=mock_repo,
        )
        result = engine.reembed_bundle("nonexistent-bundle")
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_reembed_no_md_files(self, tmp_kb, mock_chunk_store, mock_repo):
        bundle_id = "20260221-empty-a3f2"
        bundle_dir = tmp_kb / "bundles" / bundle_id
        bundle_dir.mkdir(parents=True)

        engine = ReembedEngine(
            kb_path=tmp_kb,
            kb_name="personal",
            chunk_store=mock_chunk_store,
            repo=mock_repo,
        )
        result = engine.reembed_bundle(bundle_id)
        assert result["status"] == "error"
        assert "no md files" in result["error"].lower()

    def test_reembed_reads_all_md_files(self, tmp_kb, mock_chunk_store, mock_repo):
        """여러 MD 파일이 있으면 모두 읽어서 합침."""
        bundle_id = "20260221-multi-a3f2"
        bundle_dir = tmp_kb / "bundles" / bundle_id
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "claude.md").write_text("Part 1 content", encoding="utf-8")
        (bundle_dir / "chatgpt.md").write_text("Part 2 content", encoding="utf-8")

        mock_repo.find_by_id.return_value = {
            "bundle_id": bundle_id,
            "kb": "personal",
            "platform": "claude",
            "domains": ["dev"],
            "topics": ["python"],
        }

        engine = ReembedEngine(
            kb_path=tmp_kb,
            kb_name="personal",
            chunk_store=mock_chunk_store,
            repo=mock_repo,
        )
        result = engine.reembed_bundle(bundle_id)
        assert result["status"] == "reembedded"
        # upsert should have been called with chunks from combined content
        mock_chunk_store.upsert_chunks.assert_called_once()


class TestReembedAll:
    def test_reembed_all(self, tmp_kb, mock_chunk_store, mock_repo):
        _create_bundle(tmp_kb, "20260221-aaa-a3f2")
        _create_bundle(tmp_kb, "20260222-bbb-b4e1")

        mock_repo.find_by_id.return_value = {
            "bundle_id": "dummy",
            "kb": "personal",
            "platform": "claude",
            "domains": ["dev"],
            "topics": ["python"],
        }

        engine = ReembedEngine(
            kb_path=tmp_kb,
            kb_name="personal",
            chunk_store=mock_chunk_store,
            repo=mock_repo,
        )
        stats = engine.reembed_all()
        assert stats["total"] == 2
        assert stats["reembedded"] == 2
        assert stats["errors"] == 0

    def test_reembed_all_empty_kb(self, tmp_kb, mock_chunk_store, mock_repo):
        engine = ReembedEngine(
            kb_path=tmp_kb,
            kb_name="personal",
            chunk_store=mock_chunk_store,
            repo=mock_repo,
        )
        stats = engine.reembed_all()
        assert stats["total"] == 0

    def test_reembed_all_with_progress(self, tmp_kb, mock_chunk_store, mock_repo):
        _create_bundle(tmp_kb, "20260221-aaa-a3f2")

        mock_repo.find_by_id.return_value = {
            "bundle_id": "dummy",
            "kb": "personal",
            "platform": "claude",
            "domains": [],
            "topics": [],
        }

        progress = MagicMock()
        engine = ReembedEngine(
            kb_path=tmp_kb,
            kb_name="personal",
            chunk_store=mock_chunk_store,
            repo=mock_repo,
        )
        engine.reembed_all(progress_callback=progress)
        progress.assert_called()


class TestReembedCollectionFresh:
    def test_fresh_drops_and_recreates(self, tmp_kb, mock_chunk_store, mock_repo):
        _create_bundle(tmp_kb, "20260221-aaa-a3f2")

        mock_repo.find_by_id.return_value = {
            "bundle_id": "dummy",
            "kb": "personal",
            "platform": "claude",
            "domains": [],
            "topics": [],
        }

        engine = ReembedEngine(
            kb_path=tmp_kb,
            kb_name="personal",
            chunk_store=mock_chunk_store,
            repo=mock_repo,
        )
        stats = engine.reembed_collection_fresh()
        mock_chunk_store.drop_and_recreate_collection.assert_called_once()
        assert stats["reembedded"] >= 0
