"""Tests for TopicSyncer — syncs YAML topic changes to PostgreSQL."""

from unittest.mock import MagicMock

import pytest

from pkb.vocab.syncer import TopicSyncer


@pytest.fixture
def mock_repo():
    """Mock BundleRepository."""
    return MagicMock()


@pytest.fixture
def syncer(mock_repo):
    return TopicSyncer(repo=mock_repo)


class TestSyncApprove:
    def test_approve_updates_topic_vocab(self, syncer, mock_repo):
        syncer.sync_approve("python")
        mock_repo.upsert_topic_vocab.assert_called_once_with(
            canonical="python", status="approved",
        )

    def test_approve_updates_bundle_topics(self, syncer, mock_repo):
        syncer.sync_approve("python")
        mock_repo.approve_pending_topic.assert_called_once_with("python")


class TestSyncMerge:
    def test_merge_updates_topic_vocab(self, syncer, mock_repo):
        syncer.sync_merge("py", into="python")
        mock_repo.upsert_topic_vocab.assert_called_once_with(
            canonical="py", status="merged", merged_into="python",
        )

    def test_merge_transfers_bundle_references(self, syncer, mock_repo):
        syncer.sync_merge("py", into="python")
        mock_repo.merge_topic_references.assert_called_once_with("py", "python")


class TestSyncReject:
    def test_reject_deletes_topic_vocab(self, syncer, mock_repo):
        syncer.sync_reject("spam-topic")
        mock_repo.delete_topic_vocab.assert_called_once_with("spam-topic")

    def test_reject_removes_bundle_topic_refs(self, syncer, mock_repo):
        syncer.sync_reject("spam-topic")
        mock_repo.remove_topic_from_bundles.assert_called_once_with("spam-topic")
