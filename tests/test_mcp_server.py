"""Tests for PKB MCP server tools."""

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest


class TestMCPModule:
    def test_importable(self):
        from pkb.mcp_server import create_mcp_server

        assert callable(create_mcp_server)


class TestAllToolsRegistered:
    def test_tool_names_count(self):
        from pkb.mcp_server import TOOL_NAMES

        assert len(TOOL_NAMES) == 14

    def test_all_tools_present(self):
        from pkb.mcp_server import TOOL_NAMES

        expected = {
            "pkb_search", "pkb_digest", "pkb_related", "pkb_stats",
            "pkb_ingest", "pkb_browse", "pkb_detail", "pkb_graph",
            "pkb_gaps", "pkb_claims", "pkb_timeline", "pkb_recent",
            "pkb_compare", "pkb_suggest",
        }
        assert TOOL_NAMES == expected

    def test_all_handlers_importable(self):
        from pkb.mcp_server import (
            _handle_browse,
            _handle_claims,
            _handle_compare,
            _handle_detail,
            _handle_digest,
            _handle_gaps,
            _handle_graph,
            _handle_ingest,
            _handle_recent,
            _handle_related,
            _handle_search,
            _handle_stats,
            _handle_suggest,
            _handle_timeline,
        )

        handlers = [
            _handle_browse, _handle_claims, _handle_compare,
            _handle_detail, _handle_gaps, _handle_graph,
            _handle_ingest, _handle_recent, _handle_search,
            _handle_digest, _handle_related, _handle_stats,
            _handle_suggest, _handle_timeline,
        ]
        for fn in handlers:
            assert callable(fn)


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


class TestHandleDetail:
    def test_returns_full_meta(self):
        from pkb.mcp_server import _handle_detail

        mock_repo = MagicMock()
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260228-test-abc1",
            "question": "Q?",
            "summary": "S",
            "domains": ["dev"],
            "topics": ["python"],
            "consensus": "All agree",
            "divergence": None,
            "has_synthesis": True,
        }
        mock_repo.get_responses_for_bundle.return_value = [
            {
                "platform": "claude",
                "model": "haiku",
                "key_claims": ["claim1"],
                "stance": "neutral",
                "turn_count": 5,
                "source_path": None,
            },
        ]
        mock_repo.list_relations.return_value = []

        result = _handle_detail(mock_repo, {"bundle_id": "20260228-test-abc1"})
        parsed = json.loads(result)
        assert parsed["bundle_id"] == "20260228-test-abc1"
        assert parsed["consensus"] == "All agree"
        assert len(parsed["responses"]) == 1
        assert parsed["responses"][0]["platform"] == "claude"
        assert parsed["relations"] == []

    def test_bundle_not_found(self):
        from pkb.mcp_server import _handle_detail

        mock_repo = MagicMock()
        mock_repo.get_bundle_by_id.return_value = None

        result = _handle_detail(mock_repo, {"bundle_id": "nonexistent"})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_includes_relations(self):
        from pkb.mcp_server import _handle_detail

        mock_repo = MagicMock()
        mock_repo.get_bundle_by_id.return_value = {"bundle_id": "a", "question": "Q"}
        mock_repo.get_responses_for_bundle.return_value = []
        mock_repo.list_relations.return_value = [
            {
                "source_bundle_id": "a",
                "target_bundle_id": "b",
                "relation_type": "similar",
                "score": 0.9,
            },
        ]

        result = _handle_detail(mock_repo, {"bundle_id": "a"})
        parsed = json.loads(result)
        assert len(parsed["relations"]) == 1


class TestHandleGraph:
    def test_graph_single_node(self):
        from pkb.mcp_server import _handle_graph

        mock_repo = MagicMock()
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "a",
            "question": "Q?",
            "domains": ["dev"],
            "topics": ["python"],
        }
        mock_repo.list_relations.return_value = []

        result = _handle_graph(mock_repo, {"bundle_id": "a", "depth": 1})
        parsed = json.loads(result)
        assert len(parsed["nodes"]) == 1
        assert parsed["nodes"][0]["bundle_id"] == "a"
        assert parsed["edges"] == []

    def test_graph_depth_1(self):
        from pkb.mcp_server import _handle_graph

        mock_repo = MagicMock()
        mock_repo.get_bundle_by_id.side_effect = lambda bid: {
            "bundle_id": bid,
            "question": f"Q{bid}",
            "domains": [],
            "topics": [],
        }
        mock_repo.list_relations.side_effect = lambda bid: (
            [
                {
                    "source_bundle_id": "a",
                    "target_bundle_id": "b",
                    "relation_type": "similar",
                    "score": 0.9,
                }
            ]
            if bid == "a"
            else []
        )

        result = _handle_graph(mock_repo, {"bundle_id": "a", "depth": 1})
        parsed = json.loads(result)
        assert len(parsed["nodes"]) == 1  # only "a" visited at depth 1
        assert len(parsed["edges"]) == 1

    def test_graph_depth_2(self):
        from pkb.mcp_server import _handle_graph

        mock_repo = MagicMock()
        mock_repo.get_bundle_by_id.side_effect = lambda bid: {
            "bundle_id": bid,
            "question": f"Q{bid}",
            "domains": [],
            "topics": [],
        }
        mock_repo.list_relations.side_effect = lambda bid: (
            [
                {
                    "source_bundle_id": "a",
                    "target_bundle_id": "b",
                    "relation_type": "similar",
                    "score": 0.9,
                }
            ]
            if bid == "a"
            else []
        )

        result = _handle_graph(mock_repo, {"bundle_id": "a", "depth": 2})
        parsed = json.loads(result)
        bundle_ids = [n["bundle_id"] for n in parsed["nodes"]]
        assert "a" in bundle_ids
        assert "b" in bundle_ids

    def test_graph_bundle_not_found(self):
        from pkb.mcp_server import _handle_graph

        mock_repo = MagicMock()
        mock_repo.get_bundle_by_id.return_value = None

        result = _handle_graph(mock_repo, {"bundle_id": "nonexistent"})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_graph_deduplicates_edges(self):
        from pkb.mcp_server import _handle_graph

        mock_repo = MagicMock()
        mock_repo.get_bundle_by_id.side_effect = lambda bid: {
            "bundle_id": bid,
            "question": f"Q{bid}",
            "domains": [],
            "topics": [],
        }
        # Both a->b and b->a report the same edge
        mock_repo.list_relations.side_effect = lambda bid: [
            {
                "source_bundle_id": "a",
                "target_bundle_id": "b",
                "relation_type": "similar",
                "score": 0.9,
            },
        ]

        result = _handle_graph(mock_repo, {"bundle_id": "a", "depth": 2})
        parsed = json.loads(result)
        assert len(parsed["edges"]) == 1  # deduplicated

    def test_graph_depth_capped_at_3(self):
        from pkb.mcp_server import _handle_graph

        mock_repo = MagicMock()
        # Chain: a -> b -> c -> d -> e
        chain = {"a": "b", "b": "c", "c": "d", "d": "e"}
        mock_repo.get_bundle_by_id.side_effect = lambda bid: {
            "bundle_id": bid,
            "question": f"Q{bid}",
            "domains": [],
            "topics": [],
        }
        mock_repo.list_relations.side_effect = lambda bid: (
            [
                {
                    "source_bundle_id": bid,
                    "target_bundle_id": chain[bid],
                    "relation_type": "sequel",
                    "score": 0.8,
                }
            ]
            if bid in chain
            else []
        )

        # Request depth=10, should be capped to 3
        result = _handle_graph(mock_repo, {"bundle_id": "a", "depth": 10})
        parsed = json.loads(result)
        bundle_ids = [n["bundle_id"] for n in parsed["nodes"]]
        # depth 3: visits a (depth 1), b (depth 2), c (depth 3)
        assert "a" in bundle_ids
        assert "b" in bundle_ids
        assert "c" in bundle_ids
        # d is discovered at depth 3 but visited would be depth 4 (capped)
        assert "d" not in bundle_ids


class TestHandleGaps:
    def test_returns_gap_topics(self):
        from pkb.mcp_server import _handle_gaps

        mock_analytics = MagicMock()
        mock_analytics.knowledge_gaps.return_value = [
            {"topic": "kubernetes", "count": 1},
            {"topic": "rust", "count": 2},
        ]
        result = _handle_gaps(mock_analytics, {"threshold": 3})
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["topic"] == "kubernetes"

    def test_gaps_empty(self):
        from pkb.mcp_server import _handle_gaps

        mock_analytics = MagicMock()
        mock_analytics.knowledge_gaps.return_value = []
        result = _handle_gaps(mock_analytics, {"threshold": 3, "kb": "personal"})
        parsed = json.loads(result)
        assert parsed == []

    def test_gaps_passes_params(self):
        from pkb.mcp_server import _handle_gaps

        mock_analytics = MagicMock()
        mock_analytics.knowledge_gaps.return_value = []
        _handle_gaps(mock_analytics, {"threshold": 5, "kb": "work"})
        mock_analytics.knowledge_gaps.assert_called_once_with(threshold=5, kb="work")


class TestHandleClaims:
    def test_returns_matching_claims(self):
        from pkb.mcp_server import _handle_claims

        mock_repo = MagicMock()
        mock_repo.search_claims.return_value = [
            {"bundle_id": "b1", "kb": "p", "question": "Q?", "summary": "S",
             "created_at": datetime(2026, 2, 28),
             "platform": "claude", "key_claims": ["Python is slow"], "stance": "neutral"},
        ]
        result = _handle_claims(mock_repo, {"query": "Python"})
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["key_claims"] == ["Python is slow"]

    def test_claims_empty(self):
        from pkb.mcp_server import _handle_claims

        mock_repo = MagicMock()
        mock_repo.search_claims.return_value = []
        result = _handle_claims(mock_repo, {"query": "nonexistent"})
        parsed = json.loads(result)
        assert parsed == []

    def test_claims_passes_params(self):
        from pkb.mcp_server import _handle_claims

        mock_repo = MagicMock()
        mock_repo.search_claims.return_value = []
        _handle_claims(mock_repo, {"query": "test", "kb": "work", "limit": 5})
        mock_repo.search_claims.assert_called_once_with("test", kb="work", limit=5)


class TestHandleTimeline:
    def test_timeline_chronological_order(self):
        from pkb.mcp_server import _handle_timeline

        mock_repo = MagicMock()
        mock_repo.list_bundles_by_topic.return_value = [
            {"bundle_id": "b2", "kb": "p", "question": "Q2", "summary": "S2",
             "created_at": datetime(2026, 2, 1)},
            {"bundle_id": "b1", "kb": "p", "question": "Q1", "summary": "S1",
             "created_at": datetime(2026, 1, 1)},
        ]
        result = _handle_timeline(mock_repo, {"topic": "python"})
        parsed = json.loads(result)
        assert parsed[0]["bundle_id"] == "b1"  # oldest first
        assert parsed[1]["bundle_id"] == "b2"

    def test_timeline_requires_topic(self):
        from pkb.mcp_server import _handle_timeline

        result = _handle_timeline(MagicMock(), {})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_timeline_passes_kb(self):
        from pkb.mcp_server import _handle_timeline

        mock_repo = MagicMock()
        mock_repo.list_bundles_by_topic.return_value = []
        _handle_timeline(mock_repo, {"topic": "python", "kb": "work"})
        mock_repo.list_bundles_by_topic.assert_called_once_with("python", kb="work")


class TestHandleRecent:
    def test_recent_returns_bundles(self):
        from pkb.mcp_server import _handle_recent

        mock_repo = MagicMock()
        mock_repo.list_bundles_since.return_value = [
            {"bundle_id": "b1", "kb": "p", "question": "Q", "summary": "S",
             "created_at": datetime(2026, 2, 28)},
        ]
        result = _handle_recent(mock_repo, {"days": 7})
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["bundle_id"] == "b1"

    def test_recent_empty(self):
        from pkb.mcp_server import _handle_recent

        mock_repo = MagicMock()
        mock_repo.list_bundles_since.return_value = []
        result = _handle_recent(mock_repo, {})
        parsed = json.loads(result)
        assert parsed == []

    def test_recent_passes_kb(self):
        from pkb.mcp_server import _handle_recent

        mock_repo = MagicMock()
        mock_repo.list_bundles_since.return_value = []
        _handle_recent(mock_repo, {"days": 30, "kb": "personal"})
        mock_repo.list_bundles_since.assert_called_once()


class TestHandleCompare:
    def test_compare_returns_platform_responses(self):
        from pkb.mcp_server import _handle_compare

        mock_repo = MagicMock()
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "20260228-test-abc1",
            "question": "Q?",
            "summary": "S",
            "consensus": "All agree on X",
            "divergence": "Claude emphasizes Y",
            "has_synthesis": True,
        }
        mock_repo.get_responses_for_bundle.return_value = [
            {
                "platform": "claude",
                "model": "haiku",
                "key_claims": ["claim1"],
                "stance": "positive",
                "turn_count": 5,
                "source_path": None,
            },
            {
                "platform": "chatgpt",
                "model": "gpt-4o",
                "key_claims": ["claim2"],
                "stance": "neutral",
                "turn_count": 3,
                "source_path": None,
            },
        ]

        result = _handle_compare(mock_repo, {"bundle_id": "20260228-test-abc1"})
        parsed = json.loads(result)
        assert parsed["consensus"] == "All agree on X"
        assert parsed["divergence"] == "Claude emphasizes Y"
        assert len(parsed["responses"]) == 2
        assert parsed["responses"][0]["platform"] == "claude"

    def test_compare_not_found(self):
        from pkb.mcp_server import _handle_compare

        mock_repo = MagicMock()
        mock_repo.get_bundle_by_id.return_value = None
        result = _handle_compare(mock_repo, {"bundle_id": "nonexistent"})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_compare_single_platform(self):
        from pkb.mcp_server import _handle_compare

        mock_repo = MagicMock()
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "b1",
            "question": "Q?",
            "consensus": None,
            "divergence": None,
            "has_synthesis": False,
        }
        mock_repo.get_responses_for_bundle.return_value = [
            {
                "platform": "claude",
                "model": "haiku",
                "key_claims": [],
                "stance": None,
                "turn_count": 5,
                "source_path": None,
            },
        ]
        result = _handle_compare(mock_repo, {"bundle_id": "b1"})
        parsed = json.loads(result)
        assert len(parsed["responses"]) == 1
        assert parsed["has_synthesis"] is False


class TestHandleSuggest:
    def test_suggest_with_topic(self):
        from pkb.mcp_server import _handle_suggest

        mock_analytics = MagicMock()
        mock_analytics.knowledge_gaps.return_value = [
            {"topic": "kubernetes", "count": 1},
        ]
        mock_repo = MagicMock()
        mock_repo.list_bundles_by_topic.return_value = [
            {
                "bundle_id": "b1",
                "question": "Q?",
                "summary": "S",
                "kb": "p",
                "created_at": datetime(2026, 1, 1),
            },
        ]

        result = _handle_suggest(
            mock_repo, mock_analytics, {"topic": "kubernetes"}
        )
        parsed = json.loads(result)
        assert len(parsed["gaps"]) == 1
        assert len(parsed["related_bundles"]) == 1
        assert parsed["related_bundles"][0]["bundle_id"] == "b1"

    def test_suggest_without_topic(self):
        from pkb.mcp_server import _handle_suggest

        mock_analytics = MagicMock()
        mock_analytics.knowledge_gaps.return_value = [
            {"topic": "k8s", "count": 1},
            {"topic": "rust", "count": 2},
        ]
        mock_repo = MagicMock()
        result = _handle_suggest(mock_repo, mock_analytics, {})
        parsed = json.loads(result)
        assert len(parsed["gaps"]) == 2
        assert parsed["related_bundles"] == []

    def test_suggest_limits_gaps(self):
        from pkb.mcp_server import _handle_suggest

        mock_analytics = MagicMock()
        mock_analytics.knowledge_gaps.return_value = [
            {"topic": f"t{i}", "count": 1} for i in range(20)
        ]
        mock_repo = MagicMock()
        result = _handle_suggest(mock_repo, mock_analytics, {})
        parsed = json.loads(result)
        assert len(parsed["gaps"]) == 10  # capped at 10
