"""Tests for PKB MCP server tools."""

import json
from unittest.mock import MagicMock

import pytest


class TestMCPModule:
    def test_importable(self):
        from pkb.mcp_server import create_mcp_server

        assert callable(create_mcp_server)

    def test_tool_names(self):
        from pkb.mcp_server import TOOL_NAMES

        assert "pkb_search" in TOOL_NAMES
        assert "pkb_digest" in TOOL_NAMES
        assert "pkb_related" in TOOL_NAMES
        assert "pkb_stats" in TOOL_NAMES


class TestHandleSearch:
    def test_returns_json(self):
        from datetime import datetime

        from pkb.mcp_server import _handle_search
        from pkb.search.models import BundleSearchResult

        mock_engine = MagicMock()
        mock_engine.search.return_value = [
            BundleSearchResult(
                bundle_id="20260101-test-abc1",
                question="Python async?",
                summary="async/await explained",
                domains=["dev"],
                topics=["python"],
                score=0.9,
                created_at=datetime(2026, 1, 1),
                source="both",
            ),
        ]

        result = _handle_search(mock_engine, {"query": "python"})
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["bundle_id"] == "20260101-test-abc1"


class TestHandleDigest:
    def test_digest_topic(self):
        from pkb.mcp_server import _handle_digest

        mock_repo = MagicMock()
        mock_engine = MagicMock()
        mock_router = MagicMock()
        mock_config = MagicMock()

        from pkb.digest import DigestResult

        mock_engine_inst = MagicMock()
        mock_engine_inst.digest_topic.return_value = DigestResult(
            content="Topic summary", sources=[], bundle_count=1, topic="python"
        )

        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "pkb.mcp_server.DigestEngine",
                lambda **kwargs: mock_engine_inst,
            )
            result = _handle_digest(
                mock_repo, mock_engine, mock_router, mock_config,
                {"topic": "python"},
            )
        assert "Topic summary" in result


class TestHandleRelated:
    def test_returns_json(self):
        from pkb.mcp_server import _handle_related

        mock_repo = MagicMock()
        mock_repo.list_relations.return_value = [
            {
                "source_bundle_id": "20260101-a-abc1",
                "target_bundle_id": "20260101-b-def2",
                "relation_type": "similar",
                "score": 0.85,
            },
        ]

        result = _handle_related(mock_repo, {"bundle_id": "20260101-a-abc1"})
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["related_bundle"] == "20260101-b-def2"


class TestHandleStats:
    def test_returns_counts(self):
        from pkb.mcp_server import _handle_stats

        mock_repo = MagicMock()
        mock_repo.list_all_bundle_ids.return_value = ["a", "b", "c"]
        mock_repo.count_relations.return_value = 5

        result = _handle_stats(mock_repo, {})
        parsed = json.loads(result)
        assert parsed["total_bundles"] == 3
        assert parsed["total_relations"] == 5
