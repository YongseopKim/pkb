"""Tests for DuplicateDetector."""

from unittest.mock import MagicMock

import pytest

from pkb.dedup import DuplicateDetector
from pkb.models.config import DedupConfig


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def mock_chunk_store():
    return MagicMock()


@pytest.fixture
def detector(mock_repo, mock_chunk_store):
    config = DedupConfig(threshold=0.85)
    return DuplicateDetector(
        repo=mock_repo,
        chunk_store=mock_chunk_store,
        config=config,
    )


class TestScanBundleEmptyQuestion:
    def test_empty_question_returns_empty(self, detector, mock_repo, mock_chunk_store):
        """Bundles with empty question should return empty (no TEI call)."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-md-only-abc1",
            "question": "",
            "kb": "personal",
        }

        pairs = detector.scan_bundle("20260101-md-only-abc1")
        assert pairs == []
        mock_chunk_store.search.assert_not_called()

    def test_none_question_returns_empty(self, detector, mock_repo, mock_chunk_store):
        """Bundles with None question should return empty."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-md-only-abc1",
            "question": None,
            "kb": "personal",
        }

        pairs = detector.scan_bundle("20260101-md-only-abc1")
        assert pairs == []
        mock_chunk_store.search.assert_not_called()


class TestScanBundle:
    def test_scan_bundle_finds_duplicate(self, detector, mock_repo, mock_chunk_store):
        """When a bundle's question is similar to another bundle's chunk, detect it."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-new-abc1",
            "question": "Python async 패턴은?",
            "kb": "personal",
        }
        # ChromaDB returns similar chunk from a different bundle
        mock_chunk_store.search.return_value = [
            MagicMock(
                chunk_id="20260101-old-def2_chunk_0",
                document="Python 비동기 프로그래밍 패턴",
                metadata={"bundle_id": "20260101-old-def2", "kb": "personal"},
                distance=0.10,  # very similar: 1 - 0.10 = 0.90 > 0.85
            ),
        ]
        mock_repo.list_duplicate_pairs.return_value = []

        pairs = detector.scan_bundle("20260101-new-abc1")
        assert len(pairs) == 1
        assert pairs[0]["bundle_a"] == "20260101-new-abc1"
        assert pairs[0]["bundle_b"] == "20260101-old-def2"
        assert pairs[0]["similarity"] >= 0.85

    def test_scan_bundle_ignores_self(self, detector, mock_repo, mock_chunk_store):
        """Should not report a bundle as duplicate of itself."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-test-abc1",
            "question": "테스트 질문",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = [
            MagicMock(
                chunk_id="20260101-test-abc1_chunk_0",
                document="테스트 질문 내용",
                metadata={"bundle_id": "20260101-test-abc1", "kb": "personal"},
                distance=0.0,
            ),
        ]
        mock_repo.list_duplicate_pairs.return_value = []

        pairs = detector.scan_bundle("20260101-test-abc1")
        assert len(pairs) == 0

    def test_scan_bundle_below_threshold(self, detector, mock_repo, mock_chunk_store):
        """Below threshold should not be detected."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-new-abc1",
            "question": "Python async",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = [
            MagicMock(
                chunk_id="20260101-other-xyz9_chunk_0",
                document="JavaScript frameworks overview",
                metadata={"bundle_id": "20260101-other-xyz9", "kb": "personal"},
                distance=0.50,  # 1 - 0.50 = 0.50 < 0.85
            ),
        ]
        mock_repo.list_duplicate_pairs.return_value = []

        pairs = detector.scan_bundle("20260101-new-abc1")
        assert len(pairs) == 0

    def test_scan_bundle_skips_existing_pairs(self, detector, mock_repo, mock_chunk_store):
        """Already registered pairs should be skipped."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-new-abc1",
            "question": "Python async",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = [
            MagicMock(
                chunk_id="20260101-old-def2_chunk_0",
                document="Python async patterns",
                metadata={"bundle_id": "20260101-old-def2", "kb": "personal"},
                distance=0.05,
            ),
        ]
        # Already registered
        mock_repo.list_duplicate_pairs.return_value = [
            {"id": 1, "bundle_a": "20260101-new-abc1", "bundle_b": "20260101-old-def2"},
        ]

        pairs = detector.scan_bundle("20260101-new-abc1")
        assert len(pairs) == 0


class TestScan:
    def test_scan_all_bundles(self, detector, mock_repo, mock_chunk_store):
        """scan() should iterate over all bundles."""
        mock_repo.list_all_bundle_ids.return_value = ["b1", "b2"]
        b1 = {"bundle_id": "b1", "question": "Q1", "kb": "personal"}
        b2 = {"bundle_id": "b2", "question": "Q2", "kb": "personal"}
        mock_repo.get_bundle_by_id.side_effect = [
            b1,   # scan() check for b1
            b1,   # scan_bundle() internal lookup
            b2,   # scan() check for b2
            b2,   # scan_bundle() internal lookup
        ]
        mock_chunk_store.search.return_value = []
        mock_repo.list_duplicate_pairs.return_value = []

        result = detector.scan()
        assert result["scanned"] == 2

    def test_scan_with_kb_filter(self, detector, mock_repo, mock_chunk_store):
        """scan(kb=...) should pass kb filter."""
        mock_repo.list_all_bundle_ids.return_value = []
        mock_repo.list_duplicate_pairs.return_value = []

        detector.scan(kb="work")
        mock_repo.list_all_bundle_ids.assert_called_once_with(kb="work")

    def test_scan_skips_empty_question_bundles(self, detector, mock_repo, mock_chunk_store):
        """scan() should skip bundles with empty question and track skipped count."""
        mock_repo.list_all_bundle_ids.return_value = ["empty1", "good1"]
        good_bundle = {"bundle_id": "good1", "question": "Real question", "kb": "personal"}
        mock_repo.get_bundle_by_id.side_effect = [
            # scan() checks each bundle
            {"bundle_id": "empty1", "question": "", "kb": "personal"},
            good_bundle,   # scan() check for good1
            good_bundle,   # scan_bundle -> internal lookup
        ]
        mock_chunk_store.search.return_value = []
        mock_repo.list_duplicate_pairs.return_value = []

        result = detector.scan()
        assert result["scanned"] == 2
        assert result["skipped"] == 1
        mock_chunk_store.search.assert_called_once()


class TestListPairs:
    def test_list_pairs_default(self, detector, mock_repo):
        mock_repo.list_duplicate_pairs.return_value = [
            {"id": 1, "bundle_a": "a", "bundle_b": "b", "similarity": 0.9, "status": "pending"},
        ]
        pairs = detector.list_pairs()
        assert len(pairs) == 1

    def test_list_pairs_with_status(self, detector, mock_repo):
        mock_repo.list_duplicate_pairs.return_value = []
        detector.list_pairs(status="dismissed")
        mock_repo.list_duplicate_pairs.assert_called_once_with(status="dismissed")


class TestDismissConfirm:
    def test_dismiss_pair(self, detector, mock_repo):
        detector.dismiss_pair(42)
        mock_repo.update_duplicate_status.assert_called_once_with(42, "dismissed")

    def test_confirm_pair(self, detector, mock_repo):
        detector.confirm_pair(42)
        mock_repo.update_duplicate_status.assert_called_once_with(42, "confirmed")
