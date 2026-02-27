"""Tests for PostIngestProcessor — post-ingest automation pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from pkb.models.config import DedupConfig, PostIngestConfig, RelationConfig
from pkb.post_ingest import PostIngestProcessor, PostIngestResult


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def mock_chunk_store():
    return MagicMock()


@pytest.fixture
def post_ingest_config():
    return PostIngestConfig(auto_relate=True, auto_dedup=True, gap_update=True)


@pytest.fixture
def relation_config():
    return RelationConfig(similarity_threshold=0.7, max_relations_per_bundle=20)


@pytest.fixture
def dedup_config():
    return DedupConfig(threshold=0.85)


@pytest.fixture
def processor(mock_repo, mock_chunk_store, post_ingest_config, relation_config, dedup_config):
    return PostIngestProcessor(
        repo=mock_repo,
        chunk_store=mock_chunk_store,
        config=post_ingest_config,
        relation_config=relation_config,
        dedup_config=dedup_config,
        gap_threshold=3,
    )


class TestPostIngestResult:
    def test_default_values(self):
        result = PostIngestResult()
        assert result.new_relations == 0
        assert result.new_dedup_pairs == 0
        assert result.gap_topics == []

    def test_custom_values(self):
        result = PostIngestResult(
            new_relations=5,
            new_dedup_pairs=2,
            gap_topics=["python", "docker"],
        )
        assert result.new_relations == 5
        assert result.new_dedup_pairs == 2
        assert result.gap_topics == ["python", "docker"]


class TestAutoRelate:
    def test_auto_relate_enabled_calls_scan_and_inserts(
        self, processor, mock_repo,
    ):
        """When auto_relate is enabled, scan_bundle is called and relations are inserted."""
        with patch.object(
            processor._relation_builder, "scan_bundle",
            return_value=[
                {
                    "source_bundle_id": "20260101-src-abc1",
                    "target_bundle_id": "20260101-tgt-def2",
                    "relation_type": "similar",
                    "score": 0.85,
                },
                {
                    "source_bundle_id": "20260101-src-abc1",
                    "target_bundle_id": "20260101-tgt-ghi3",
                    "relation_type": "related",
                    "score": 0.75,
                },
            ],
        ):
            count = processor._auto_relate("20260101-src-abc1")

        assert count == 2
        assert mock_repo.insert_relation.call_count == 2
        mock_repo.insert_relation.assert_any_call(
            "20260101-src-abc1", "20260101-tgt-def2", "similar", 0.85,
        )
        mock_repo.insert_relation.assert_any_call(
            "20260101-src-abc1", "20260101-tgt-ghi3", "related", 0.75,
        )

    def test_auto_relate_disabled_skips(
        self, mock_repo, mock_chunk_store, relation_config, dedup_config,
    ):
        """When auto_relate is False, scan_bundle is NOT called."""
        config = PostIngestConfig(auto_relate=False, auto_dedup=True, gap_update=True)
        proc = PostIngestProcessor(
            repo=mock_repo,
            chunk_store=mock_chunk_store,
            config=config,
            relation_config=relation_config,
            dedup_config=dedup_config,
        )
        result = proc.process("20260101-src-abc1")
        assert result.new_relations == 0

    def test_auto_relate_empty_results(self, processor):
        """When scan_bundle returns empty list, count is 0."""
        with patch.object(
            processor._relation_builder, "scan_bundle", return_value=[],
        ):
            count = processor._auto_relate("20260101-src-abc1")
        assert count == 0

    def test_auto_relate_exception_logged_and_returns_zero(self, processor):
        """Exception in auto_relate should be caught, logged, and return 0."""
        with patch.object(
            processor._relation_builder, "scan_bundle",
            side_effect=RuntimeError("DB connection failed"),
        ):
            count = processor._auto_relate("20260101-src-abc1")
        assert count == 0


class TestAutoDedup:
    def test_auto_dedup_enabled_calls_scan_and_inserts(
        self, processor, mock_repo,
    ):
        """When auto_dedup is enabled, scan_bundle is called and pairs are inserted."""
        with patch.object(
            processor._dedup_detector, "scan_bundle",
            return_value=[
                {
                    "bundle_a": "20260101-src-abc1",
                    "bundle_b": "20260101-dup-xyz9",
                    "similarity": 0.92,
                },
            ],
        ):
            count = processor._auto_dedup("20260101-src-abc1")

        assert count == 1
        mock_repo.insert_duplicate_pair.assert_called_once_with(
            "20260101-src-abc1", "20260101-dup-xyz9", 0.92,
        )

    def test_auto_dedup_disabled_skips(
        self, mock_repo, mock_chunk_store, relation_config, dedup_config,
    ):
        """When auto_dedup is False, scan_bundle is NOT called."""
        config = PostIngestConfig(auto_relate=True, auto_dedup=False, gap_update=True)
        proc = PostIngestProcessor(
            repo=mock_repo,
            chunk_store=mock_chunk_store,
            config=config,
            relation_config=relation_config,
            dedup_config=dedup_config,
        )
        result = proc.process("20260101-src-abc1")
        assert result.new_dedup_pairs == 0

    def test_auto_dedup_empty_results(self, processor):
        """When scan_bundle returns empty list, count is 0."""
        with patch.object(
            processor._dedup_detector, "scan_bundle", return_value=[],
        ):
            count = processor._auto_dedup("20260101-src-abc1")
        assert count == 0

    def test_auto_dedup_exception_logged_and_returns_zero(self, processor):
        """Exception in auto_dedup should be caught, logged, and return 0."""
        with patch.object(
            processor._dedup_detector, "scan_bundle",
            side_effect=RuntimeError("ChromaDB timeout"),
        ):
            count = processor._auto_dedup("20260101-src-abc1")
        assert count == 0


class TestGapUpdate:
    def test_gap_update_detects_topics_below_threshold(self, processor, mock_repo):
        """Topics with bundle count below gap_threshold should be returned."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "topics": ["python", "docker", "kubernetes"],
        }
        # python: 5 bundles (above threshold 3), docker: 2 (below), kubernetes: 1 (below)
        mock_repo.count_bundles_for_topics.return_value = {
            "python": 5,
            "docker": 2,
            "kubernetes": 1,
        }

        gaps = processor._gap_update("20260101-src-abc1")
        assert "docker" in gaps
        assert "kubernetes" in gaps
        assert "python" not in gaps

    def test_gap_update_with_comma_separated_topics(self, processor, mock_repo):
        """Topics as comma-separated string should be handled."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "topics": "python, docker, kubernetes",
        }
        mock_repo.count_bundles_for_topics.return_value = {
            "python": 5,
            "docker": 1,
        }

        gaps = processor._gap_update("20260101-src-abc1")
        assert "docker" in gaps
        # kubernetes not in count_bundles_for_topics result => count=0 < threshold=3
        assert "kubernetes" in gaps
        assert "python" not in gaps

    def test_gap_update_disabled_returns_empty(
        self, mock_repo, mock_chunk_store, relation_config, dedup_config,
    ):
        """When gap_update is False, returns empty list."""
        config = PostIngestConfig(auto_relate=True, auto_dedup=True, gap_update=False)
        proc = PostIngestProcessor(
            repo=mock_repo,
            chunk_store=mock_chunk_store,
            config=config,
            relation_config=relation_config,
            dedup_config=dedup_config,
        )
        result = proc.process("20260101-src-abc1")
        assert result.gap_topics == []

    def test_gap_update_bundle_not_found(self, processor, mock_repo):
        """When bundle not found, return empty list."""
        mock_repo.get_bundle_by_id.return_value = None
        gaps = processor._gap_update("nonexistent")
        assert gaps == []

    def test_gap_update_no_topics(self, processor, mock_repo):
        """Bundle with no topics returns empty list."""
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-src-abc1",
            "topics": [],
        }
        gaps = processor._gap_update("20260101-src-abc1")
        assert gaps == []

    def test_gap_update_exception_logged_and_returns_empty(self, processor, mock_repo):
        """Exception in gap_update should be caught, logged, and return []."""
        mock_repo.get_bundle_by_id.side_effect = RuntimeError("DB error")
        gaps = processor._gap_update("20260101-src-abc1")
        assert gaps == []


class TestProcess:
    def test_process_combines_all_subtasks(self, processor, mock_repo):
        """process() should run all three sub-tasks and combine results."""
        with (
            patch.object(
                processor._relation_builder, "scan_bundle",
                return_value=[
                    {
                        "source_bundle_id": "b1",
                        "target_bundle_id": "b2",
                        "relation_type": "similar",
                        "score": 0.80,
                    },
                ],
            ),
            patch.object(
                processor._dedup_detector, "scan_bundle",
                return_value=[
                    {
                        "bundle_a": "b1",
                        "bundle_b": "b3",
                        "similarity": 0.90,
                    },
                ],
            ),
        ):
            mock_repo.get_bundle_by_id.return_value = {
                "bundle_id": "b1",
                "topics": ["python", "docker"],
            }
            mock_repo.count_bundles_for_topics.return_value = {
                "python": 10,
                "docker": 1,
            }

            result = processor.process("b1")

        assert result.new_relations == 1
        assert result.new_dedup_pairs == 1
        assert result.gap_topics == ["docker"]

    def test_process_with_all_disabled(
        self, mock_repo, mock_chunk_store, relation_config, dedup_config,
    ):
        """When all flags disabled, process returns zero results."""
        config = PostIngestConfig(auto_relate=False, auto_dedup=False, gap_update=False)
        proc = PostIngestProcessor(
            repo=mock_repo,
            chunk_store=mock_chunk_store,
            config=config,
            relation_config=relation_config,
            dedup_config=dedup_config,
        )

        result = proc.process("b1")
        assert result.new_relations == 0
        assert result.new_dedup_pairs == 0
        assert result.gap_topics == []

    def test_process_subtask_failure_does_not_crash(self, processor, mock_repo):
        """If one sub-task fails, others still run and process() doesn't raise."""
        # auto_relate fails
        with (
            patch.object(
                processor._relation_builder, "scan_bundle",
                side_effect=RuntimeError("relation scan failed"),
            ),
            patch.object(
                processor._dedup_detector, "scan_bundle",
                return_value=[
                    {
                        "bundle_a": "b1",
                        "bundle_b": "b3",
                        "similarity": 0.90,
                    },
                ],
            ),
        ):
            mock_repo.get_bundle_by_id.return_value = {
                "bundle_id": "b1",
                "topics": ["python"],
            }
            mock_repo.count_bundles_for_topics.return_value = {"python": 10}

            result = processor.process("b1")

        # auto_relate failed => 0, but others succeed
        assert result.new_relations == 0
        assert result.new_dedup_pairs == 1
        assert result.gap_topics == []

    def test_process_gap_update_failure_does_not_crash(self, processor, mock_repo):
        """If gap_update fails, process() still returns results from other sub-tasks."""
        with (
            patch.object(
                processor._relation_builder, "scan_bundle",
                return_value=[],
            ),
            patch.object(
                processor._dedup_detector, "scan_bundle",
                return_value=[],
            ),
        ):
            mock_repo.get_bundle_by_id.side_effect = RuntimeError("DB crashed")
            result = processor.process("b1")

        assert result.new_relations == 0
        assert result.new_dedup_pairs == 0
        assert result.gap_topics == []
