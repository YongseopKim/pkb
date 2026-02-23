"""Tests for RelationBuilder — knowledge graph edge detection."""

from unittest.mock import MagicMock

import pytest

from pkb.models.config import RelationConfig
from pkb.relations import RelationBuilder


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def mock_chunk_store():
    return MagicMock()


@pytest.fixture
def builder(mock_repo, mock_chunk_store):
    config = RelationConfig(similarity_threshold=0.7, max_relations_per_bundle=20)
    return RelationBuilder(
        repo=mock_repo,
        chunk_store=mock_chunk_store,
        config=config,
    )


class TestFindSimilar:
    def test_finds_similar_bundle(self, builder, mock_repo, mock_chunk_store):
        """When embedding similarity > threshold, detect similar relation."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "question": "Python async 패턴은?",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = [
            MagicMock(
                metadata={"bundle_id": "20260101-tgt-def2", "kb": "personal"},
                distance=0.20,  # similarity = 0.80 > 0.7
            ),
        ]

        relations = builder.find_similar("20260101-src-abc1")
        assert len(relations) == 1
        assert relations[0]["source_bundle_id"] == "20260101-src-abc1"
        assert relations[0]["target_bundle_id"] == "20260101-tgt-def2"
        assert relations[0]["relation_type"] == "similar"
        assert relations[0]["score"] >= 0.7

    def test_ignores_self(self, builder, mock_repo, mock_chunk_store):
        """Should not relate a bundle to itself."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "question": "테스트",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = [
            MagicMock(
                metadata={"bundle_id": "20260101-src-abc1", "kb": "personal"},
                distance=0.0,
            ),
        ]

        relations = builder.find_similar("20260101-src-abc1")
        assert len(relations) == 0

    def test_below_threshold(self, builder, mock_repo, mock_chunk_store):
        """Below threshold should not be detected."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "question": "Python",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = [
            MagicMock(
                metadata={"bundle_id": "20260101-other-xyz9", "kb": "personal"},
                distance=0.50,  # similarity = 0.50 < 0.7
            ),
        ]

        relations = builder.find_similar("20260101-src-abc1")
        assert len(relations) == 0

    def test_respects_max_relations(self, builder, mock_repo, mock_chunk_store):
        """Should not exceed max_relations_per_bundle."""
        builder._max_relations = 2
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "question": "테스트",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = [
            MagicMock(
                metadata={"bundle_id": f"20260101-tgt-{i:04d}", "kb": "personal"},
                distance=0.10,
            )
            for i in range(5)
        ]

        relations = builder.find_similar("20260101-src-abc1")
        assert len(relations) <= 2

    def test_none_bundle_returns_empty(self, builder, mock_repo):
        """Non-existent bundle should return empty."""
        mock_repo.get_bundle_by_id.return_value = None
        relations = builder.find_similar("nonexistent")
        assert relations == []

    def test_deduplicates_by_bundle_id(self, builder, mock_repo, mock_chunk_store):
        """Multiple chunks from same bundle should produce one relation."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "question": "Python",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = [
            MagicMock(
                metadata={"bundle_id": "20260101-tgt-def2", "kb": "personal"},
                distance=0.15,
            ),
            MagicMock(
                metadata={"bundle_id": "20260101-tgt-def2", "kb": "personal"},
                distance=0.20,
            ),
        ]

        relations = builder.find_similar("20260101-src-abc1")
        assert len(relations) == 1
        # Should keep the best (highest similarity = lowest distance)
        assert relations[0]["score"] == pytest.approx(0.85, abs=0.01)


class TestFindRelatedByTopics:
    def test_finds_shared_topics(self, builder, mock_repo):
        """Bundles sharing topics should be related."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "question": "Python async",
            "kb": "personal",
        }
        mock_repo.find_bundles_sharing_topics.return_value = [
            {"bundle_id": "20260101-tgt-def2", "shared_count": 3, "total_topics": 4},
            {"bundle_id": "20260101-tgt-ghi3", "shared_count": 1, "total_topics": 5},
        ]

        relations = builder.find_related_by_topics("20260101-src-abc1")
        assert len(relations) >= 1
        assert all(r["relation_type"] == "related" for r in relations)

    def test_none_bundle_returns_empty(self, builder, mock_repo):
        mock_repo.get_bundle_by_id.return_value = None
        relations = builder.find_related_by_topics("nonexistent")
        assert relations == []

    def test_low_overlap_filtered_out(self, builder, mock_repo):
        """Bundles with very low topic overlap should be filtered out."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "question": "Python",
            "kb": "personal",
        }
        mock_repo.find_bundles_sharing_topics.return_value = [
            {"bundle_id": "20260101-tgt-def2", "shared_count": 1, "total_topics": 10},
        ]
        # 1/10 = 0.1 < 0.3 threshold
        relations = builder.find_related_by_topics("20260101-src-abc1")
        assert len(relations) == 0


class TestScanBundle:
    def test_scan_bundle_combines_types(self, builder, mock_repo, mock_chunk_store):
        """scan_bundle should combine similar + related relations."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "question": "Python async",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = [
            MagicMock(
                metadata={"bundle_id": "20260101-tgt-def2", "kb": "personal"},
                distance=0.15,
            ),
        ]
        mock_repo.find_bundles_sharing_topics.return_value = [
            {"bundle_id": "20260101-tgt-ghi3", "shared_count": 2, "total_topics": 3},
        ]

        relations = builder.scan_bundle("20260101-src-abc1")
        types = {r["relation_type"] for r in relations}
        assert "similar" in types
        assert "related" in types


class TestScanAll:
    def test_scan_all(self, builder, mock_repo, mock_chunk_store):
        """scan() should process all bundles and return stats."""
        mock_repo.list_all_bundle_ids.return_value = ["b1", "b2"]
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "b1",
            "question": "Q1",
            "kb": "personal",
        }
        mock_chunk_store.search.return_value = []
        mock_repo.find_bundles_sharing_topics.return_value = []

        result = builder.scan(kb="personal")
        assert result["scanned"] == 2
        mock_repo.list_all_bundle_ids.assert_called_once_with(kb="personal")
