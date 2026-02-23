"""Tests for web analytics dashboard routes."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from pkb.web.app import create_app
from pkb.web.deps import AppState


@pytest.fixture
def mock_state():
    repo = MagicMock()
    chunk_store = MagicMock()
    search_engine = MagicMock()
    return AppState(
        repo=repo,
        chunk_store=chunk_store,
        search_engine=search_engine,
    )


@pytest.fixture
def client(mock_state):
    app = create_app(mock_state)
    return TestClient(app)


class TestAnalyticsDashboard:
    def test_dashboard_renders(self, client, mock_state):
        mock_state.repo.list_all_bundle_ids.return_value = ["b1", "b2"]
        mock_state.repo.count_bundles_by_domain.return_value = [
            {"domain": "dev", "count": 2},
        ]
        mock_state.repo.count_bundles_by_topic.return_value = []
        mock_state.repo.count_relations.return_value = 5
        resp = client.get("/analytics")
        assert resp.status_code == 200
        assert "Analytics" in resp.text

    def test_dashboard_contains_overview(self, client, mock_state):
        mock_state.repo.list_all_bundle_ids.return_value = ["b1"]
        mock_state.repo.count_bundles_by_domain.return_value = []
        mock_state.repo.count_bundles_by_topic.return_value = []
        mock_state.repo.count_relations.return_value = 0
        resp = client.get("/analytics")
        assert resp.status_code == 200


class TestAnalyticsAPIDomains:
    def test_domains_json(self, client, mock_state):
        mock_state.repo.count_bundles_by_domain.return_value = [
            {"domain": "dev", "count": 10},
        ]
        resp = client.get("/analytics/api/domains")
        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["domain"] == "dev"


class TestAnalyticsAPITopics:
    def test_topics_json(self, client, mock_state):
        mock_state.repo.count_bundles_by_topic.return_value = [
            {"topic": "python", "count": 5},
        ]
        resp = client.get("/analytics/api/topics")
        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["topic"] == "python"


class TestAnalyticsAPITrend:
    def test_trend_json(self, client, mock_state):
        mock_state.repo.count_bundles_by_month.return_value = [
            {"month": "2026-02", "count": 3},
        ]
        resp = client.get("/analytics/api/trend")
        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["month"] == "2026-02"


class TestAnalyticsAPIPlatforms:
    def test_platforms_json(self, client, mock_state):
        mock_state.repo.count_responses_by_platform.return_value = [
            {"platform": "claude", "count": 20},
        ]
        resp = client.get("/analytics/api/platforms")
        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["platform"] == "claude"


class TestAnalyticsAPIGaps:
    def test_gaps_json(self, client, mock_state):
        mock_state.repo.count_bundles_by_topic.return_value = [
            {"topic": "python", "count": 10},
            {"topic": "k8s", "count": 1},
        ]
        resp = client.get("/analytics/api/gaps")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["topic"] == "k8s"


class TestBaseTemplateNavLink:
    def test_analytics_nav_link_exists(self, client, mock_state):
        mock_state.repo.list_all_bundle_ids.return_value = []
        resp = client.get("/")
        assert resp.status_code == 200
        assert 'href="/analytics"' in resp.text
