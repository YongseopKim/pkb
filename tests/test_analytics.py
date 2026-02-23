"""Tests for AnalyticsEngine module."""

from unittest.mock import MagicMock

import pytest

from pkb.analytics import AnalyticsEngine


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def engine(mock_repo):
    return AnalyticsEngine(repo=mock_repo)


class TestAnalyticsEngineConstruction:
    def test_creates_with_repo(self, engine):
        assert engine is not None

    def test_has_repo(self, engine, mock_repo):
        assert engine._repo is mock_repo


class TestDomainDistribution:
    def test_delegates_to_repo(self, engine, mock_repo):
        mock_repo.count_bundles_by_domain.return_value = [
            {"domain": "dev", "count": 10},
        ]
        result = engine.domain_distribution()
        mock_repo.count_bundles_by_domain.assert_called_once_with(kb=None)
        assert result == [{"domain": "dev", "count": 10}]

    def test_with_kb(self, engine, mock_repo):
        mock_repo.count_bundles_by_domain.return_value = []
        engine.domain_distribution(kb="work")
        mock_repo.count_bundles_by_domain.assert_called_once_with(kb="work")


class TestTopicHeatmap:
    def test_returns_top_n(self, engine, mock_repo):
        mock_repo.count_bundles_by_topic.return_value = [
            {"topic": f"t{i}", "count": 20 - i} for i in range(25)
        ]
        result = engine.topic_heatmap(top_n=10)
        assert len(result) == 10

    def test_default_top_20(self, engine, mock_repo):
        mock_repo.count_bundles_by_topic.return_value = [
            {"topic": f"t{i}", "count": 30 - i} for i in range(30)
        ]
        result = engine.topic_heatmap()
        assert len(result) == 20


class TestTemporalTrend:
    def test_delegates_with_months(self, engine, mock_repo):
        mock_repo.count_bundles_by_month.return_value = [
            {"month": "2026-02", "count": 5},
        ]
        result = engine.temporal_trend(months=3)
        mock_repo.count_bundles_by_month.assert_called_once_with(kb=None, months=3)
        assert result == [{"month": "2026-02", "count": 5}]


class TestPlatformDistribution:
    def test_delegates_to_repo(self, engine, mock_repo):
        mock_repo.count_responses_by_platform.return_value = [
            {"platform": "claude", "count": 20},
        ]
        result = engine.platform_distribution()
        mock_repo.count_responses_by_platform.assert_called_once_with(kb=None)
        assert result == [{"platform": "claude", "count": 20}]


class TestKnowledgeGaps:
    def test_filters_below_threshold(self, engine, mock_repo):
        mock_repo.count_bundles_by_topic.return_value = [
            {"topic": "python", "count": 15},
            {"topic": "docker", "count": 2},
            {"topic": "k8s", "count": 1},
        ]
        result = engine.knowledge_gaps(threshold=3)
        assert len(result) == 2
        assert result[0]["topic"] == "docker"
        assert result[1]["topic"] == "k8s"

    def test_default_threshold_3(self, engine, mock_repo):
        mock_repo.count_bundles_by_topic.return_value = [
            {"topic": "a", "count": 3},
            {"topic": "b", "count": 2},
        ]
        result = engine.knowledge_gaps()
        assert len(result) == 1
        assert result[0]["topic"] == "b"


class TestOverview:
    def test_returns_overview_dict(self, engine, mock_repo):
        mock_repo.count_bundles_by_domain.return_value = [
            {"domain": "dev", "count": 5},
            {"domain": "data", "count": 3},
        ]
        mock_repo.count_bundles_by_topic.return_value = [
            {"topic": "python", "count": 10},
        ]
        mock_repo.list_all_bundle_ids.return_value = ["b1", "b2", "b3"]
        mock_repo.count_relations = MagicMock(return_value=7)
        result = engine.overview()
        assert result["total_bundles"] == 3
        assert result["total_relations"] == 7
        assert result["domain_count"] == 2
        assert result["topic_count"] == 1
