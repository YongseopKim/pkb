"""Tests for compare view routes."""

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
    return AppState(repo=repo, chunk_store=chunk_store, search_engine=search_engine)


@pytest.fixture
def client(mock_state):
    app = create_app(mock_state)
    return TestClient(app)


class TestCompareRoute:
    def test_compare_page_renders(self, client, mock_state):
        """Compare page should render with bundle_id."""
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
            {"platform": "claude", "model": "claude-3", "turn_count": 5,
             "key_claims": ["claim1"], "stance": "positive", "source_path": None},
            {"platform": "chatgpt", "model": "gpt-4", "turn_count": 3,
             "key_claims": ["claim2"], "stance": "neutral", "source_path": None},
        ]
        resp = client.get("/compare/20260101-test-abc1")
        assert resp.status_code == 200
        assert "Compare" in resp.text or "compare" in resp.text.lower()
        assert "claude" in resp.text
        assert "chatgpt" in resp.text

    def test_compare_not_found(self, client, mock_state):
        """Compare page returns 404 for missing bundle."""
        mock_state.repo.get_bundle_by_id.return_value = None
        resp = client.get("/compare/nonexistent")
        assert resp.status_code == 404

    def test_compare_api_claims(self, client, mock_state):
        """Compare API should return claims by platform."""
        mock_state.repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-test-abc1",
            "question": "Test", "summary": "Sum",
            "kb": "p", "domains": "dev", "topics": "py", "created_at": None,
        }
        mock_state.repo.get_responses_for_bundle.return_value = [
            {"platform": "claude", "model": "claude-3", "turn_count": 5,
             "key_claims": ["A is true", "B is false"], "stance": "positive",
             "source_path": None},
            {"platform": "chatgpt", "model": "gpt-4", "turn_count": 3,
             "key_claims": ["A is true", "C is maybe"], "stance": "neutral",
             "source_path": None},
        ]
        resp = client.get("/compare/api/20260101-test-abc1")
        assert resp.status_code == 200
        data = resp.json()
        assert "platforms" in data
        assert len(data["platforms"]) == 2
        assert "consensus" in data

    def test_compare_api_not_found(self, client, mock_state):
        """Compare API returns 404 for missing bundle."""
        mock_state.repo.get_bundle_by_id.return_value = None
        resp = client.get("/compare/api/nonexistent")
        assert resp.status_code == 404

    def test_compare_api_consensus_detection(self, client, mock_state):
        """Consensus claims appear in 2+ platforms."""
        mock_state.repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-test-abc1",
            "question": "Test", "summary": "Sum",
            "kb": "p", "domains": "dev", "topics": "py", "created_at": None,
        }
        mock_state.repo.get_responses_for_bundle.return_value = [
            {"platform": "claude", "model": "claude-3", "turn_count": 5,
             "key_claims": ["shared claim", "claude only"], "stance": "positive",
             "source_path": None},
            {"platform": "chatgpt", "model": "gpt-4", "turn_count": 3,
             "key_claims": ["shared claim", "chatgpt only"], "stance": "neutral",
             "source_path": None},
        ]
        resp = client.get("/compare/api/20260101-test-abc1")
        data = resp.json()
        assert "shared claim" in data["consensus"]
        assert "claude only" not in data["consensus"]
        assert "chatgpt only" not in data["consensus"]

    def test_compare_page_single_response(self, client, mock_state):
        """Compare page should render even with single platform."""
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
            {"platform": "claude", "model": "claude-3", "turn_count": 5,
             "key_claims": ["claim1"], "stance": "positive", "source_path": None},
        ]
        resp = client.get("/compare/20260101-test-abc1")
        assert resp.status_code == 200
        assert "claude" in resp.text
