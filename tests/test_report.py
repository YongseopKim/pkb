"""Tests for ReportGenerator module."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from pkb.report import ReportGenerator


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.list_bundles_since.return_value = [
        {
            "bundle_id": "20260220-test-bundle-a1b2",
            "kb": "personal",
            "question": "테스트 질문",
            "summary": "테스트 요약",
            "created_at": datetime(2026, 2, 20, tzinfo=timezone.utc),
        },
    ]
    return repo


@pytest.fixture
def mock_analytics():
    analytics = MagicMock()
    analytics.domain_distribution.return_value = [
        {"domain": "dev", "count": 5},
        {"domain": "data", "count": 3},
    ]
    analytics.topic_heatmap.return_value = [
        {"topic": "python", "count": 10},
        {"topic": "docker", "count": 4},
    ]
    analytics.knowledge_gaps.return_value = [
        {"topic": "k8s", "count": 1},
    ]
    return analytics


@pytest.fixture
def generator(mock_repo, mock_analytics):
    return ReportGenerator(repo=mock_repo, analytics=mock_analytics)


class TestReportGeneratorConstruction:
    def test_creates_with_deps(self, generator):
        assert generator is not None


class TestWeeklyReport:
    def test_returns_markdown_string(self, generator):
        result = generator.weekly()
        assert isinstance(result, str)
        assert "주간" in result

    def test_contains_bundle_list(self, generator):
        result = generator.weekly()
        assert "20260220-test-bundle-a1b2" in result

    def test_contains_domain_section(self, generator):
        result = generator.weekly()
        assert "dev" in result

    def test_with_kb_filter(self, generator, mock_repo):
        generator.weekly(kb="personal")
        mock_repo.list_bundles_since.assert_called_once()
        call_kwargs = mock_repo.list_bundles_since.call_args[1]
        assert call_kwargs.get("kb") == "personal"


class TestMonthlyReport:
    def test_returns_markdown_string(self, generator):
        result = generator.monthly()
        assert isinstance(result, str)
        assert "월간" in result

    def test_contains_knowledge_gaps(self, generator):
        result = generator.monthly()
        assert "k8s" in result

    def test_with_kb_filter(self, generator, mock_repo):
        generator.monthly(kb="work")
        call_kwargs = mock_repo.list_bundles_since.call_args[1]
        assert call_kwargs.get("kb") == "work"
