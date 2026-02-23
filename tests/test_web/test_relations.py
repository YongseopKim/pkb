"""Tests for relations web routes."""

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


class TestRelationsRoutes:
    def test_relations_list(self, client, mock_state):
        mock_state.repo.list_all_relations.return_value = [
            {
                "id": 1,
                "source_bundle_id": "20260101-a-abc1",
                "target_bundle_id": "20260101-b-def2",
                "relation_type": "similar",
                "score": 0.85,
                "created_at": None,
            },
        ]
        mock_state.repo.count_relations.return_value = 1

        resp = client.get("/relations")
        assert resp.status_code == 200
        assert "20260101-a-abc1" in resp.text

    def test_relations_detail(self, client, mock_state):
        mock_state.repo.list_relations.return_value = [
            {
                "id": 1,
                "source_bundle_id": "20260101-a-abc1",
                "target_bundle_id": "20260101-b-def2",
                "relation_type": "similar",
                "score": 0.85,
                "created_at": None,
            },
        ]
        mock_state.repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260101-a-abc1",
            "question": "Test question",
            "summary": "Test summary",
            "kb": "personal",
            "domains": "dev",
            "topics": "python",
            "created_at": None,
        }

        resp = client.get("/relations/20260101-a-abc1")
        assert resp.status_code == 200
        assert "20260101-b-def2" in resp.text

    def test_relations_graph_json(self, client, mock_state):
        mock_state.repo.list_all_relations.return_value = [
            {
                "id": 1,
                "source_bundle_id": "a",
                "target_bundle_id": "b",
                "relation_type": "similar",
                "score": 0.85,
                "created_at": None,
            },
        ]

        resp = client.get("/relations/api/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["edges"]) == 1

    def test_relations_list_empty(self, client, mock_state):
        mock_state.repo.list_all_relations.return_value = []
        mock_state.repo.count_relations.return_value = 0

        resp = client.get("/relations")
        assert resp.status_code == 200
