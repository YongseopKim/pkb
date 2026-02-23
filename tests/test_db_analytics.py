"""Tests for BundleRepository analytics aggregate methods."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from pkb.db.postgres import BundleRepository
from pkb.models.config import PostgresConfig


@pytest.fixture
def mock_conn():
    """Mock psycopg connection."""
    conn = MagicMock()
    conn.execute = MagicMock()
    return conn


@pytest.fixture
def repo(mock_conn):
    """BundleRepository with mocked connection."""
    config = PostgresConfig(host="localhost", password="test")
    with patch("pkb.db.postgres.psycopg.connect", return_value=mock_conn):
        r = BundleRepository(config)
    return r


# ─── count_bundles_by_domain ─────────────────────────────

class TestCountBundlesByDomain:
    def test_method_exists(self, repo):
        assert hasattr(repo, "count_bundles_by_domain")

    def test_returns_list_of_dicts(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = [
            ("dev", 10),
            ("data", 5),
        ]
        result = repo.count_bundles_by_domain()
        assert result == [
            {"domain": "dev", "count": 10},
            {"domain": "data", "count": 5},
        ]

    def test_with_kb_filter(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = [("dev", 3)]
        result = repo.count_bundles_by_domain(kb="personal")
        assert result == [{"domain": "dev", "count": 3}]
        sql, params = mock_conn.execute.call_args[0]
        assert "kb = %s" in sql
        assert "personal" in params

    def test_empty_result(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = []
        result = repo.count_bundles_by_domain()
        assert result == []


# ─── count_bundles_by_topic ──────────────────────────────

class TestCountBundlesByTopic:
    def test_method_exists(self, repo):
        assert hasattr(repo, "count_bundles_by_topic")

    def test_returns_list_of_dicts(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = [
            ("python", 15),
            ("docker", 8),
        ]
        result = repo.count_bundles_by_topic()
        assert result == [
            {"topic": "python", "count": 15},
            {"topic": "docker", "count": 8},
        ]

    def test_with_kb_filter(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = [("python", 5)]
        result = repo.count_bundles_by_topic(kb="work")
        assert result == [{"topic": "python", "count": 5}]
        sql, params = mock_conn.execute.call_args[0]
        assert "kb = %s" in sql
        assert "work" in params

    def test_empty_result(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = []
        result = repo.count_bundles_by_topic()
        assert result == []


# ─── count_bundles_by_month ──────────────────────────────

class TestCountBundlesByMonth:
    def test_method_exists(self, repo):
        assert hasattr(repo, "count_bundles_by_month")

    def test_returns_list_of_dicts(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = [
            ("2026-02", 12),
            ("2026-01", 8),
        ]
        result = repo.count_bundles_by_month()
        assert result == [
            {"month": "2026-02", "count": 12},
            {"month": "2026-01", "count": 8},
        ]

    def test_custom_months(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = [("2026-02", 5)]
        repo.count_bundles_by_month(months=3)
        sql, params = mock_conn.execute.call_args[0]
        assert 3 in params

    def test_with_kb_filter(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.count_bundles_by_month(kb="personal", months=12)
        sql, params = mock_conn.execute.call_args[0]
        assert "personal" in params

    def test_empty_result(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = []
        result = repo.count_bundles_by_month()
        assert result == []


# ─── count_responses_by_platform ─────────────────────────

class TestCountResponsesByPlatform:
    def test_method_exists(self, repo):
        assert hasattr(repo, "count_responses_by_platform")

    def test_returns_list_of_dicts(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = [
            ("claude", 20),
            ("chatgpt", 15),
        ]
        result = repo.count_responses_by_platform()
        assert result == [
            {"platform": "claude", "count": 20},
            {"platform": "chatgpt", "count": 15},
        ]

    def test_with_kb_filter(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = [("claude", 10)]
        repo.count_responses_by_platform(kb="personal")
        sql, params = mock_conn.execute.call_args[0]
        assert "kb = %s" in sql
        assert "personal" in params

    def test_empty_result(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = []
        result = repo.count_responses_by_platform()
        assert result == []


# ─── list_bundles_since ──────────────────────────────────

class TestListBundlesSince:
    def test_method_exists(self, repo):
        assert hasattr(repo, "list_bundles_since")

    def test_returns_list_of_dicts(self, repo, mock_conn):
        ts = datetime(2026, 2, 20, tzinfo=timezone.utc)
        mock_conn.execute.return_value.fetchall.return_value = [
            ("b1", "personal", "Question 1", "Summary 1", ts),
        ]
        result = repo.list_bundles_since(ts)
        assert len(result) == 1
        assert result[0]["bundle_id"] == "b1"
        assert result[0]["question"] == "Question 1"
        assert result[0]["created_at"] == ts

    def test_with_kb_filter(self, repo, mock_conn):
        ts = datetime(2026, 2, 1, tzinfo=timezone.utc)
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.list_bundles_since(ts, kb="work")
        sql, params = mock_conn.execute.call_args[0]
        assert "kb = %s" in sql
        assert "work" in params

    def test_empty_result(self, repo, mock_conn):
        ts = datetime(2026, 2, 23, tzinfo=timezone.utc)
        mock_conn.execute.return_value.fetchall.return_value = []
        result = repo.list_bundles_since(ts)
        assert result == []
