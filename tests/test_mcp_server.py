"""Tests for PKB MCP server tools."""

import json
from datetime import datetime
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


class TestHandleIngest:
    def test_ingest_returns_bundle_info(self, tmp_path):
        from pkb.mcp_server import _handle_ingest

        # Create a real file so Path.exists() returns True
        test_file = tmp_path / "test.jsonl"
        test_file.write_text("{}")

        mock_result = {
            "bundle_id": "20260228-test-abc1",
            "summary": "Test summary",
            "domains": ["dev"],
            "topics": ["python"],
        }
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file.return_value = mock_result

        result = _handle_ingest(mock_pipeline, {"file_path": str(test_file)})
        parsed = json.loads(result)
        assert parsed["bundle_id"] == "20260228-test-abc1"

    def test_ingest_file_not_found(self):
        from pkb.mcp_server import _handle_ingest

        result = _handle_ingest(None, {"file_path": "/nonexistent/file.jsonl"})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_ingest_returns_none(self, tmp_path):
        from pkb.mcp_server import _handle_ingest

        test_file = tmp_path / "test.jsonl"
        test_file.write_text("{}")

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file.return_value = None

        result = _handle_ingest(mock_pipeline, {"file_path": str(test_file)})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_ingest_skip_result(self, tmp_path):
        from pkb.mcp_server import _handle_ingest

        test_file = tmp_path / "test.jsonl"
        test_file.write_text("{}")

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file.return_value = {"status": "skip_parse_error"}

        result = _handle_ingest(mock_pipeline, {"file_path": str(test_file)})
        parsed = json.loads(result)
        assert parsed["status"] == "skip_parse_error"


class TestHandleBrowse:
    def test_browse_by_domain(self):
        from pkb.mcp_server import _handle_browse

        mock_repo = MagicMock()
        mock_repo.list_bundles_by_domain.return_value = [
            {
                "bundle_id": "b1",
                "kb": "p",
                "question": "Q?",
                "summary": "S",
                "created_at": datetime(2026, 2, 28),
            },
        ]
        result = _handle_browse(mock_repo, {"domain": "dev"})
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["bundle_id"] == "b1"
        mock_repo.list_bundles_by_domain.assert_called_once_with("dev", kb=None)

    def test_browse_by_topic(self):
        from pkb.mcp_server import _handle_browse

        mock_repo = MagicMock()
        mock_repo.list_bundles_by_topic.return_value = [
            {
                "bundle_id": "b1",
                "kb": "p",
                "question": "Q?",
                "summary": "S",
                "created_at": datetime(2026, 2, 28),
            },
        ]
        result = _handle_browse(mock_repo, {"topic": "python"})
        parsed = json.loads(result)
        assert len(parsed) == 1
        mock_repo.list_bundles_by_topic.assert_called_once_with("python", kb=None)

    def test_browse_recent_days(self):
        from pkb.mcp_server import _handle_browse

        mock_repo = MagicMock()
        mock_repo.list_bundles_since.return_value = []
        result = _handle_browse(mock_repo, {"days": 7})
        parsed = json.loads(result)
        assert parsed == []

    def test_browse_with_limit(self):
        from pkb.mcp_server import _handle_browse

        mock_repo = MagicMock()
        mock_repo.list_bundles_by_domain.return_value = [
            {"bundle_id": f"b{i}"} for i in range(30)
        ]
        result = _handle_browse(mock_repo, {"domain": "dev", "limit": 5})
        parsed = json.loads(result)
        assert len(parsed) == 5

    def test_browse_requires_filter(self):
        from pkb.mcp_server import _handle_browse

        result = _handle_browse(MagicMock(), {})
        parsed = json.loads(result)
        assert "error" in parsed
