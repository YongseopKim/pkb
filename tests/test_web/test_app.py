"""Tests for web app factory and routes."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from pkb.web.app import create_app
from pkb.web.deps import AppState


@pytest.fixture
def mock_state():
    """Create a mock AppState."""
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
    """FastAPI test client."""
    app = create_app(mock_state)
    return TestClient(app)


class TestDashboard:
    def test_dashboard_renders(self, client, mock_state):
        mock_state.repo.list_all_bundle_ids.return_value = ["b1", "b2", "b3"]
        resp = client.get("/")
        assert resp.status_code == 200
        assert "PKB" in resp.text
        assert "3" in resp.text  # total bundles

    def test_dashboard_empty(self, client, mock_state):
        mock_state.repo.list_all_bundle_ids.return_value = []
        resp = client.get("/")
        assert resp.status_code == 200


class TestBundleRoutes:
    def test_bundle_list(self, client, mock_state):
        mock_state.repo.list_all_bundle_ids.return_value = ["20260101-test-abc1"]
        mock_state.repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-test-abc1",
            "question": "Test question",
            "summary": "Test summary",
            "kb": "personal",
            "domains": "dev",
            "topics": "python",
            "created_at": "2026-01-01T00:00:00Z",
        }
        resp = client.get("/bundles")
        assert resp.status_code == 200
        assert "20260101-test-abc1" in resp.text

    def test_bundle_detail(self, client, mock_state):
        mock_state.repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-test-abc1",
            "question": "Test question",
            "summary": "Test summary",
            "kb": "personal",
            "domains": "dev",
            "topics": "python",
            "created_at": "2026-01-01T00:00:00Z",
        }
        resp = client.get("/bundles/20260101-test-abc1")
        assert resp.status_code == 200
        assert "Test summary" in resp.text

    def test_bundle_detail_no_question_label(self, client, mock_state):
        """bundle detail 페이지에 'Question:' 라벨이 없어야 함."""
        mock_state.repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-test-abc1",
            "question": "Test question",
            "summary": "Test summary",
            "kb": "personal",
            "domains": "dev",
            "topics": "python",
            "created_at": "2026-01-01T00:00:00Z",
        }
        resp = client.get("/bundles/20260101-test-abc1")
        assert resp.status_code == 200
        assert "Question:" not in resp.text

    def test_bundle_detail_not_found(self, client, mock_state):
        mock_state.repo.get_bundle_by_id.return_value = None
        resp = client.get("/bundles/nonexistent")
        assert resp.status_code == 404

    def test_bundle_delete(self, client, mock_state):
        resp = client.post("/bundles/20260101-test-abc1/delete", follow_redirects=False)
        assert resp.status_code == 303
        mock_state.repo.delete_bundle.assert_called_once_with("20260101-test-abc1")


class TestSearchRoutes:
    def test_search_page_no_query(self, client):
        resp = client.get("/search")
        assert resp.status_code == 200
        assert "Search" in resp.text

    def test_search_with_query(self, client, mock_state):
        from datetime import datetime, timezone

        from pkb.search.models import BundleSearchResult
        mock_state.search_engine.search.return_value = [
            BundleSearchResult(
                bundle_id="20260101-test-abc1",
                question="Test question",
                summary="Test summary",
                domains=["dev"],
                topics=["python"],
                score=0.85,
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                source="both",
            ),
        ]
        resp = client.get("/search?q=test")
        assert resp.status_code == 200
        assert "0.85" in resp.text

    def test_search_no_question_in_html(self, client, mock_state):
        """search 결과에 question 줄이 없어야 함."""
        from datetime import datetime, timezone

        from pkb.search.models import BundleSearchResult
        mock_state.search_engine.search.return_value = [
            BundleSearchResult(
                bundle_id="20260101-test-abc1",
                question="Test question",
                summary="Test summary",
                domains=["dev"],
                topics=["python"],
                score=0.85,
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                source="both",
            ),
        ]
        resp = client.get("/search?q=test")
        assert resp.status_code == 200
        assert 'class="question"' not in resp.text


class TestDuplicateRoutes:
    def test_duplicates_list(self, client, mock_state):
        mock_state.repo.list_duplicate_pairs.return_value = [
            {"id": 1, "bundle_a": "a", "bundle_b": "b", "similarity": 0.9, "status": "pending"},
        ]
        resp = client.get("/duplicates")
        assert resp.status_code == 200
        assert "0.90" in resp.text

    def test_duplicate_dismiss(self, client, mock_state):
        resp = client.post("/duplicates/1/dismiss", follow_redirects=False)
        assert resp.status_code == 303
        mock_state.repo.update_duplicate_status.assert_called_once_with(1, "dismissed")


class TestSettingsRoute:
    def test_settings_page(self, client, monkeypatch, tmp_path):
        monkeypatch.setenv("PKB_HOME", str(tmp_path))
        config_file = tmp_path / "config.yaml"
        config_file.write_text("knowledge_bases: []\n")
        resp = client.get("/settings")
        assert resp.status_code == 200
        assert "knowledge_bases" in resp.text


class TestChatRoute:
    def test_chat_page_placeholder(self, client):
        resp = client.get("/chat")
        assert resp.status_code == 200
        assert "Chat" in resp.text


class TestDashboardEnhanced:
    def test_dashboard_has_recent_activity(self, client, mock_state):
        """Dashboard should show recent bundles with metadata."""
        mock_state.repo.list_all_bundle_ids.return_value = ["b1", "b2"]
        mock_state.repo.list_bundles_since.return_value = [
            {"bundle_id": "b1", "question": "Q1", "kb": "personal", "created_at": "2026-02-28"},
        ]
        mock_state.repo.count_bundles_by_domain.return_value = [
            {"domain": "dev", "count": 5},
        ]
        mock_state.repo.count_bundles_by_topic.return_value = [
            {"topic": "python", "count": 2},
        ]
        mock_state.repo.count_relations.return_value = 3
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Recent Activity" in resp.text or "recent" in resp.text.lower()

    def test_dashboard_has_knowledge_gaps(self, client, mock_state):
        """Dashboard should show knowledge gap cards."""
        mock_state.repo.list_all_bundle_ids.return_value = ["b1"]
        mock_state.repo.list_bundles_since.return_value = []
        mock_state.repo.count_bundles_by_domain.return_value = [
            {"domain": "dev", "count": 5},
        ]
        mock_state.repo.count_bundles_by_topic.return_value = [
            {"topic": "k8s", "count": 1},
        ]
        mock_state.repo.count_relations.return_value = 0
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Knowledge Gaps" in resp.text or "gaps" in resp.text.lower()

    def test_dashboard_has_domain_chart(self, client, mock_state):
        """Dashboard should include domain distribution chart."""
        mock_state.repo.list_all_bundle_ids.return_value = ["b1"]
        mock_state.repo.list_bundles_since.return_value = []
        mock_state.repo.count_bundles_by_domain.return_value = []
        mock_state.repo.count_bundles_by_topic.return_value = []
        mock_state.repo.count_relations.return_value = 0
        resp = client.get("/")
        assert resp.status_code == 200
        assert "domainMiniChart" in resp.text or "chart.js" in resp.text.lower()


class TestBundleDetailCompareLink:
    def test_bundle_detail_has_compare_link(self, client, mock_state):
        """Bundle detail should have Compare button when multi-platform."""
        mock_state.repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-test-abc1",
            "question": "Test question",
            "summary": "Test summary",
            "kb": "personal",
            "domains": "dev",
            "topics": "python",
            "created_at": None,
        }
        mock_state.repo.get_responses_for_bundle.return_value = [
            {"platform": "claude", "model": "m1", "turn_count": 3,
             "key_claims": [], "stance": "", "source_path": None},
            {"platform": "chatgpt", "model": "m2", "turn_count": 2,
             "key_claims": [], "stance": "", "source_path": None},
        ]
        resp = client.get("/bundles/20260101-test-abc1")
        assert resp.status_code == 200
        assert "/compare/20260101-test-abc1" in resp.text

    def test_bundle_detail_no_compare_single_platform(self, client, mock_state):
        """Bundle detail should NOT have Compare button for single platform."""
        mock_state.repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-test-abc1",
            "question": "Test question",
            "summary": "Test summary",
            "kb": "personal",
            "domains": "dev",
            "topics": "python",
            "created_at": None,
        }
        mock_state.repo.get_responses_for_bundle.return_value = [
            {"platform": "claude", "model": "m1", "turn_count": 3,
             "key_claims": [], "stance": "", "source_path": None},
        ]
        resp = client.get("/bundles/20260101-test-abc1")
        assert resp.status_code == 200
        assert "/compare/20260101-test-abc1" not in resp.text


class TestChatUIEnhanced:
    def test_chat_page_has_context_panel(self, client):
        """Chat page should have context sidebar panel."""
        resp = client.get("/chat")
        assert resp.status_code == 200
        assert "context-panel" in resp.text

    def test_chat_send_returns_sources(self, client, mock_state):
        """Chat send should return source references in response."""
        from unittest.mock import MagicMock as _MagicMock

        from pkb.chat.models import ChatResponse

        mock_engine = _MagicMock()
        mock_engine.ask.return_value = ChatResponse(
            content="GIL is a mutex.",
            sources=[
                {"bundle_id": "20260101-python-abc1", "question": "Python GIL?", "score": 0.9},
            ],
        )
        mock_state.chat_engine = mock_engine
        resp = client.post("/chat/send", data={"message": "What is GIL?"})
        assert resp.status_code == 200
        assert "20260101-python-abc1" in resp.text


class TestStyleConsistency:
    def test_static_css_loads(self, client, mock_state):
        """Static CSS should be accessible and contain new component styles."""
        resp = client.get("/static/style.css")
        assert resp.status_code == 200
        assert "dashboard-grid" in resp.text
        assert "compare-grid" in resp.text
        assert "chat-layout" in resp.text
