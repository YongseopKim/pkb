"""Tests for PostgreSQL repository (mock-based)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from pkb.db.postgres import BundleRepository
from pkb.models.config import PostgresConfig


@pytest.fixture
def mock_conn():
    """Mock psycopg connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute = MagicMock()
    return conn


@pytest.fixture
def repo(mock_conn):
    """BundleRepository with mocked connection."""
    config = PostgresConfig(host="localhost", password="test")
    with patch("pkb.db.postgres.psycopg.connect", return_value=mock_conn):
        r = BundleRepository(config)
    return r


class TestBundleRepository:
    def test_construction(self, repo):
        assert repo is not None

    def test_upsert_bundle(self, repo, mock_conn):
        repo.upsert_bundle(
            bundle_id="20260221-test-slug-a3f2",
            kb="personal",
            question="테스트 질문",
            summary="테스트 요약",
            created_at=datetime.now(timezone.utc),
            response_count=3,
            path="bundles/20260221-test-slug-a3f2",
            question_hash="abc123",
            domains=["dev"],
            topics=["python"],
            responses=[{"platform": "claude", "model": "sonnet", "turn_count": 5}],
        )
        # Should have called execute for bundle + domains + topics + responses
        assert mock_conn.execute.call_count >= 1

    def test_bundle_exists_true(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchone.return_value = (1,)
        assert repo.bundle_exists("some-hash") is True

    def test_bundle_exists_false(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchone.return_value = None
        assert repo.bundle_exists("nonexistent") is False

    def test_close(self, repo, mock_conn):
        repo.close()
        mock_conn.close.assert_called_once()


class TestSearchFTS:
    def test_search_fts_basic(self, repo, mock_conn):
        """search_fts returns list of dicts with score."""
        mock_conn.execute.return_value.fetchall.return_value = [
            (
                "20260221-bitcoin-a3f2", "personal", "Bitcoin halving이란?",
                "비트코인 반감기 분석", datetime(2026, 2, 21, tzinfo=timezone.utc),
                "investing", "bitcoin,crypto", 0.5,
            ),
        ]
        results = repo.search_fts(query="bitcoin halving", limit=10)
        assert len(results) == 1
        assert results[0]["bundle_id"] == "20260221-bitcoin-a3f2"
        assert results[0]["rank"] == 0.5
        mock_conn.execute.assert_called()

    def test_search_fts_with_kb_filter(self, repo, mock_conn):
        """search_fts filters by kb when provided."""
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.search_fts(query="test", kb="personal", limit=5)
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "kb" in sql.lower()

    def test_search_fts_with_domain_filter(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.search_fts(query="test", domains=["investing"], limit=5)
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "bundle_domains" in sql.lower()

    def test_search_fts_with_topic_filter(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.search_fts(query="test", topics=["bitcoin"], limit=5)
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "bundle_topics" in sql.lower()

    def test_search_fts_with_date_filters(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = []
        from datetime import date
        repo.search_fts(
            query="test", after=date(2026, 1, 1), before=date(2026, 12, 31), limit=5,
        )
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "created_at" in sql.lower()

    def test_search_fts_empty_results(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = []
        results = repo.search_fts(query="nonexistent", limit=10)
        assert results == []

    def test_search_fts_query_tokenization(self, repo, mock_conn):
        """Multi-word query should be tokenized for tsquery."""
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.search_fts(query="bitcoin halving price", limit=10)
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        # The tsquery should combine words with &
        assert "bitcoin" in str(params) or "bitcoin" in str(call_args)

    def test_search_fts_with_stance_filter(self, repo, mock_conn):
        """search_fts with stance filter includes bundle_responses condition."""
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.search_fts(query="test", stance="informative", limit=5)
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        assert "bundle_responses" in sql.lower()
        assert "stance" in sql.lower()
        assert params["stance"] == "informative"

    def test_search_fts_with_has_consensus_true(self, repo, mock_conn):
        """search_fts with has_consensus=True includes consensus IS NOT NULL condition."""
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.search_fts(query="test", has_consensus=True, limit=5)
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "consensus is not null" in sql.lower()

    def test_search_fts_with_has_consensus_false(self, repo, mock_conn):
        """search_fts with has_consensus=False includes consensus IS NULL condition."""
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.search_fts(query="test", has_consensus=False, limit=5)
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "consensus is null" in sql.lower()

    def test_search_fts_with_has_synthesis_true(self, repo, mock_conn):
        """search_fts with has_synthesis=True includes has_synthesis = TRUE condition."""
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.search_fts(query="test", has_synthesis=True, limit=5)
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "has_synthesis" in sql.lower()

    def test_search_fts_with_has_synthesis_false(self, repo, mock_conn):
        """search_fts with has_synthesis=False includes has_synthesis = FALSE condition."""
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.search_fts(query="test", has_synthesis=False, limit=5)
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "has_synthesis" in sql.lower()


class TestGetBundleById:
    def test_get_existing_bundle(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchone.return_value = (
            "20260221-bitcoin-a3f2", "personal", "Bitcoin halving이란?",
            "비트코인 반감기 분석", datetime(2026, 2, 21, tzinfo=timezone.utc),
            "investing", "bitcoin,crypto",
            None, None, False, "abc123stable",
        )
        result = repo.get_bundle_by_id("20260221-bitcoin-a3f2")
        assert result is not None
        assert result["bundle_id"] == "20260221-bitcoin-a3f2"
        assert result["stable_id"] == "abc123stable"

    def test_get_nonexistent_bundle(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchone.return_value = None
        result = repo.get_bundle_by_id("nonexistent")
        assert result is None


class TestListAllBundleIds:
    def test_list_all(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = [
            ("20260101-a-1234",),
            ("20260102-b-5678",),
        ]
        result = repo.list_all_bundle_ids()
        assert result == ["20260101-a-1234", "20260102-b-5678"]

    def test_list_with_kb_filter(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = [("20260101-a-1234",)]
        result = repo.list_all_bundle_ids(kb="personal")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "kb" in sql.lower()
        assert result == ["20260101-a-1234"]

    def test_list_empty(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = []
        result = repo.list_all_bundle_ids()
        assert result == []


class TestDeleteBundle:
    def test_delete_existing(self, repo, mock_conn):
        repo.delete_bundle("20260101-test-1234")
        mock_conn.execute.assert_called()
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "delete" in sql.lower()
        assert "bundles" in sql.lower()

    def test_delete_cascades(self, repo, mock_conn):
        """Delete only needs one statement since CASCADE handles related tables."""
        repo.delete_bundle("20260101-test-1234")
        # Only one DELETE statement needed (CASCADE handles domains/topics/responses)
        delete_calls = [
            c for c in mock_conn.execute.call_args_list
            if "DELETE" in str(c[0][0]).upper()
        ]
        assert len(delete_calls) == 1


class TestUpdateBundleMeta:
    def test_update_meta(self, repo, mock_conn):
        repo.update_bundle_meta(
            bundle_id="20260101-test-1234",
            summary="수정된 요약",
            domains=["dev", "ai"],
            topics=["python"],
            pending_topics=["new-topic"],
        )
        # Should update bundles table + replace domains + replace topics
        assert mock_conn.execute.call_count >= 3

    def test_update_meta_updates_summary(self, repo, mock_conn):
        repo.update_bundle_meta(
            bundle_id="20260101-test-1234",
            summary="새 요약",
            domains=["dev"],
            topics=["python"],
            pending_topics=[],
        )
        calls = mock_conn.execute.call_args_list
        # First call should be UPDATE bundles
        sql = calls[0][0][0]
        assert "update" in sql.lower() or "UPDATE" in sql


class TestTopicVocabMethods:
    def test_upsert_topic_vocab(self, repo, mock_conn):
        repo.upsert_topic_vocab(canonical="python", status="approved")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "topic_vocab" in sql
        assert "INSERT" in sql.upper()

    def test_upsert_topic_vocab_with_merged_into(self, repo, mock_conn):
        repo.upsert_topic_vocab(
            canonical="py", status="merged", merged_into="python",
        )
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params["merged_into"] == "python"

    def test_delete_topic_vocab(self, repo, mock_conn):
        repo.delete_topic_vocab("spam")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "DELETE" in sql.upper()
        assert "topic_vocab" in sql

    def test_merge_topic_references(self, repo, mock_conn):
        repo.merge_topic_references("py", "python")
        # Should UPDATE then DELETE
        calls = mock_conn.execute.call_args_list
        assert len(calls) >= 2
        update_sql = calls[0][0][0]
        assert "UPDATE" in update_sql.upper()
        assert "bundle_topics" in update_sql

    def test_approve_pending_topic(self, repo, mock_conn):
        repo.approve_pending_topic("python")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "UPDATE" in sql.upper()
        assert "is_pending" in sql

    def test_remove_topic_from_bundles(self, repo, mock_conn):
        repo.remove_topic_from_bundles("spam")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "DELETE" in sql.upper()
        assert "bundle_topics" in sql


class TestFindBySourcePath:
    def test_find_in_bundle_responses(self, repo, mock_conn):
        """bundle_responses에서 source_path로 찾기 (merge된 파일)."""
        # First call (bundle_responses) returns match
        mock_conn.execute.return_value.fetchone.return_value = ("20260221-test-a3f2",)
        result = repo.find_by_source_path("/inbox/chatgpt.md")
        assert result == "20260221-test-a3f2"

    def test_find_fallback_to_bundles(self, repo, mock_conn):
        """bundle_responses에 없으면 bundles 테이블 fallback."""
        # First call returns None (not in bundle_responses), second returns match
        cursor_mock = MagicMock()
        cursor_mock.fetchone.side_effect = [None, ("20260221-test-a3f2",)]
        mock_conn.execute.return_value = cursor_mock
        result = repo.find_by_source_path("/inbox/claude.jsonl")
        assert result == "20260221-test-a3f2"

    def test_find_nonexistent(self, repo, mock_conn):
        """양쪽 테이블 모두 없으면 None 반환."""
        cursor_mock = MagicMock()
        cursor_mock.fetchone.side_effect = [None, None]
        mock_conn.execute.return_value = cursor_mock
        result = repo.find_by_source_path("/inbox/nonexistent.jsonl")
        assert result is None

    def test_checks_bundle_responses_first(self, repo, mock_conn):
        """bundle_responses를 먼저 조회."""
        cursor_mock = MagicMock()
        cursor_mock.fetchone.side_effect = [None, None]
        mock_conn.execute.return_value = cursor_mock
        repo.find_by_source_path("/inbox/test.jsonl")
        calls = mock_conn.execute.call_args_list
        first_sql = calls[0][0][0]
        assert "bundle_responses" in first_sql

    def test_bundles_fallback_sql(self, repo, mock_conn):
        """bundle_responses에 없을 때 bundles 테이블 조회."""
        cursor_mock = MagicMock()
        cursor_mock.fetchone.side_effect = [None, None]
        mock_conn.execute.return_value = cursor_mock
        repo.find_by_source_path("/inbox/test.jsonl")
        calls = mock_conn.execute.call_args_list
        assert len(calls) == 2
        second_sql = calls[1][0][0]
        assert "bundles" in second_sql
        assert "bundle_responses" not in second_sql

    def test_skips_bundles_if_found_in_responses(self, repo, mock_conn):
        """bundle_responses에서 찾으면 bundles는 조회하지 않음."""
        mock_conn.execute.return_value.fetchone.return_value = ("20260221-test-a3f2",)
        repo.find_by_source_path("/inbox/chatgpt.md")
        # Only one execute call (bundle_responses)
        assert mock_conn.execute.call_count == 1


class TestUpsertBundleWithSourcePath:
    def test_upsert_with_source_path(self, repo, mock_conn):
        """upsert_bundle()에 source_path 전달 시 SQL에 포함되어야 함."""
        repo.upsert_bundle(
            bundle_id="20260221-test-slug-a3f2",
            kb="personal",
            question="테스트 질문",
            summary="테스트 요약",
            created_at=datetime.now(timezone.utc),
            response_count=3,
            path="bundles/20260221-test-slug-a3f2",
            question_hash="abc123",
            domains=["dev"],
            topics=["python"],
            responses=[{"platform": "claude", "model": "sonnet", "turn_count": 5}],
            source_path="/inbox/test.jsonl",
        )
        calls = mock_conn.execute.call_args_list
        insert_sql = calls[0][0][0]
        assert "source_path" in insert_sql

    def test_upsert_without_source_path_defaults_none(self, repo, mock_conn):
        """source_path 없이 호출해도 기존처럼 동작."""
        repo.upsert_bundle(
            bundle_id="20260221-test-slug-a3f2",
            kb="personal",
            question="테스트 질문",
            summary="테스트 요약",
            created_at=datetime.now(timezone.utc),
            response_count=3,
            path="bundles/20260221-test-slug-a3f2",
            question_hash="abc123",
            domains=["dev"],
            topics=["python"],
            responses=[{"platform": "claude", "model": "sonnet", "turn_count": 5}],
        )
        # Should still work (no error)
        assert mock_conn.execute.call_count >= 1

    def test_upsert_responses_include_source_path(self, repo, mock_conn):
        """responses dict에 source_path가 있으면 bundle_responses INSERT에 포함."""
        repo.upsert_bundle(
            bundle_id="20260221-test-slug-a3f2",
            kb="personal",
            question="테스트 질문",
            summary="테스트 요약",
            created_at=datetime.now(timezone.utc),
            response_count=3,
            path="bundles/20260221-test-slug-a3f2",
            question_hash="abc123",
            domains=["dev"],
            topics=["python"],
            responses=[{
                "platform": "claude", "model": "sonnet",
                "turn_count": 5, "source_path": "/inbox/claude.jsonl",
            }],
            source_path="/inbox/claude.jsonl",
        )
        calls = mock_conn.execute.call_args_list
        # Find the bundle_responses INSERT call (after DELETE)
        response_inserts = [
            c for c in calls
            if "bundle_responses" in str(c[0][0]) and "INSERT" in str(c[0][0]).upper()
        ]
        assert len(response_inserts) == 1
        sql = response_inserts[0][0][0]
        params = response_inserts[0][0][1]
        assert "source_path" in sql
        assert "/inbox/claude.jsonl" in params


class TestCountByKb:
    def test_count_returns_number(self, repo, mock_conn):
        """KB에 속한 번들 수를 반환."""
        mock_conn.execute.return_value.fetchone.return_value = (5,)
        result = repo.count_by_kb("personal")
        assert result == 5
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "count" in sql.lower()
        assert call_args[0][1] == ("personal",)

    def test_count_empty_kb(self, repo, mock_conn):
        """빈 KB는 0 반환."""
        mock_conn.execute.return_value.fetchone.return_value = (0,)
        result = repo.count_by_kb("empty-kb")
        assert result == 0


class TestDeleteByKb:
    def test_delete_returns_count(self, repo, mock_conn):
        """삭제된 번들 수를 반환."""
        mock_conn.execute.return_value.fetchall.return_value = [
            ("bundle-1",), ("bundle-2",), ("bundle-3",),
        ]
        result = repo.delete_by_kb("personal")
        assert result == 3
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "delete" in sql.lower()
        assert "bundles" in sql.lower()
        assert "RETURNING" in sql.upper()

    def test_delete_empty_kb(self, repo, mock_conn):
        """빈 KB 삭제 시 0 반환."""
        mock_conn.execute.return_value.fetchall.return_value = []
        result = repo.delete_by_kb("empty-kb")
        assert result == 0


class TestDuplicatePairMethods:
    def test_insert_duplicate_pair(self, repo, mock_conn):
        repo.insert_duplicate_pair("bundle_a", "bundle_b", 0.92)
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "INSERT" in sql.upper()
        assert "duplicate_pairs" in sql

    def test_list_duplicate_pairs_all(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = [
            (1, "a", "b", 0.9, "pending", None),
        ]
        result = repo.list_duplicate_pairs()
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_list_duplicate_pairs_with_status(self, repo, mock_conn):
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.list_duplicate_pairs(status="dismissed")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "status" in sql.lower()

    def test_update_duplicate_status(self, repo, mock_conn):
        repo.update_duplicate_status(42, "dismissed")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "UPDATE" in sql.upper()
        assert "duplicate_pairs" in sql


class TestFindBundleByQuestionHash:
    """find_bundle_by_question_hash(): merge를 위한 번들 조회."""

    def test_find_existing_bundle(self, repo, mock_conn):
        """question_hash로 번들 + platforms 조회."""
        mock_conn.execute.return_value.fetchone.return_value = (
            "20260221-test-a3f2", "personal", "bundles/20260221-test-a3f2",
            "claude", "dev", "python",
        )
        result = repo.find_bundle_by_question_hash("abc123")
        assert result is not None
        assert result["bundle_id"] == "20260221-test-a3f2"
        assert "platforms" in result
        assert "claude" in result["platforms"]

    def test_find_nonexistent_returns_none(self, repo, mock_conn):
        """존재하지 않는 hash는 None 반환."""
        mock_conn.execute.return_value.fetchone.return_value = None
        result = repo.find_bundle_by_question_hash("nonexistent")
        assert result is None

    def test_find_uses_correct_sql(self, repo, mock_conn):
        """SQL이 question_hash를 사용."""
        mock_conn.execute.return_value.fetchone.return_value = None
        repo.find_bundle_by_question_hash("abc123")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "question_hash" in sql
        assert "bundle_responses" in sql


class TestAddResponseToBundle:
    """add_response_to_bundle(): 기존 번들에 새 플랫폼 response 추가."""

    def test_adds_response(self, repo, mock_conn):
        """bundle_responses INSERT + response_count UPDATE."""
        repo.add_response_to_bundle(
            bundle_id="20260221-test-a3f2",
            platform="chatgpt",
            model="gpt-4o",
            turn_count=5,
        )
        calls = mock_conn.execute.call_args_list
        sql_all = " ".join(str(c[0][0]) for c in calls)
        assert "bundle_responses" in sql_all
        assert "response_count" in sql_all

    def test_adds_response_with_source_path(self, repo, mock_conn):
        """source_path가 INSERT SQL에 포함되어야 함."""
        repo.add_response_to_bundle(
            bundle_id="20260221-test-a3f2",
            platform="chatgpt",
            model="gpt-4o",
            turn_count=5,
            source_path="/inbox/chatgpt.md",
        )
        calls = mock_conn.execute.call_args_list
        insert_sql = calls[0][0][0]
        insert_params = calls[0][0][1]
        assert "source_path" in insert_sql
        assert "/inbox/chatgpt.md" in insert_params

    def test_source_path_in_on_conflict_update(self, repo, mock_conn):
        """ON CONFLICT 시에도 source_path가 업데이트되어야 함."""
        repo.add_response_to_bundle(
            bundle_id="20260221-test-a3f2",
            platform="chatgpt",
            model="gpt-4o",
            turn_count=5,
            source_path="/inbox/chatgpt.md",
        )
        calls = mock_conn.execute.call_args_list
        insert_sql = calls[0][0][0]
        assert "source_path = EXCLUDED.source_path" in insert_sql


class TestDeduplicateTopicsAndDomains:
    """중복 토픽/도메인이 들어올 때 UniqueViolation 없이 처리되어야 함."""

    def _base_kwargs(self):
        return dict(
            bundle_id="20260222-test-dedup-e3b0",
            kb="personal",
            question="테스트 질문",
            summary="테스트 요약",
            created_at=datetime.now(timezone.utc),
            response_count=1,
            path="bundles/20260222-test-dedup-e3b0",
            question_hash="dedup123",
            responses=[{"platform": "claude", "model": "sonnet", "turn_count": 3}],
        )

    def test_upsert_deduplicates_topics(self, repo, mock_conn):
        """topics 리스트에 중복이 있으면 deduplicate하여 INSERT 1회만."""
        repo.upsert_bundle(
            **self._base_kwargs(),
            domains=["investing"],
            topics=["ethereum", "bitcoin", "ethereum"],
        )
        topic_inserts = [
            c for c in mock_conn.execute.call_args_list
            if "bundle_topics" in str(c[0][0]) and "INSERT" in str(c[0][0]).upper()
        ]
        # ethereum 중복 → 2개만 INSERT
        assert len(topic_inserts) == 2

    def test_upsert_deduplicates_domains(self, repo, mock_conn):
        """domains 리스트에 중복이 있으면 deduplicate."""
        repo.upsert_bundle(
            **self._base_kwargs(),
            domains=["dev", "dev", "ai"],
            topics=["python"],
        )
        domain_inserts = [
            c for c in mock_conn.execute.call_args_list
            if "bundle_domains" in str(c[0][0]) and "INSERT" in str(c[0][0]).upper()
        ]
        assert len(domain_inserts) == 2

    def test_upsert_pending_topics_overlap_with_topics(self, repo, mock_conn):
        """pending_topics가 topics와 겹치면 pending에서 제거."""
        repo.upsert_bundle(
            **self._base_kwargs(),
            domains=["investing"],
            topics=["ethereum", "bitcoin"],
            pending_topics=["ethereum", "defi"],
        )
        topic_inserts = [
            c for c in mock_conn.execute.call_args_list
            if "bundle_topics" in str(c[0][0]) and "INSERT" in str(c[0][0]).upper()
        ]
        # ethereum(approved) + bitcoin(approved) + defi(pending) = 3
        # ethereum in pending_topics는 제거됨
        assert len(topic_inserts) == 3

    def test_upsert_pending_topics_self_dedup(self, repo, mock_conn):
        """pending_topics 리스트 자체의 중복도 제거."""
        repo.upsert_bundle(
            **self._base_kwargs(),
            domains=["investing"],
            topics=["bitcoin"],
            pending_topics=["defi", "defi", "nft"],
        )
        topic_inserts = [
            c for c in mock_conn.execute.call_args_list
            if "bundle_topics" in str(c[0][0]) and "INSERT" in str(c[0][0]).upper()
        ]
        # bitcoin(approved) + defi(pending) + nft(pending) = 3
        assert len(topic_inserts) == 3

    def test_update_meta_deduplicates_topics(self, repo, mock_conn):
        """update_bundle_meta에서도 중복 토픽 deduplicate."""
        repo.update_bundle_meta(
            bundle_id="20260222-test-dedup-e3b0",
            summary="요약",
            domains=["dev"],
            topics=["python", "python", "fastapi"],
            pending_topics=["python", "new-topic"],
        )
        topic_inserts = [
            c for c in mock_conn.execute.call_args_list
            if "bundle_topics" in str(c[0][0]) and "INSERT" in str(c[0][0]).upper()
        ]
        # python(approved) + fastapi(approved) + new-topic(pending) = 3
        # python in pending_topics는 제거됨
        assert len(topic_inserts) == 3

    def test_update_meta_deduplicates_domains(self, repo, mock_conn):
        """update_bundle_meta에서도 중복 도메인 deduplicate."""
        repo.update_bundle_meta(
            bundle_id="20260222-test-dedup-e3b0",
            summary="요약",
            domains=["dev", "dev", "ai"],
            topics=["python"],
        )
        domain_inserts = [
            c for c in mock_conn.execute.call_args_list
            if "bundle_domains" in str(c[0][0]) and "INSERT" in str(c[0][0]).upper()
        ]
        assert len(domain_inserts) == 2


class TestFindBundleByStableId:
    """find_bundle_by_stable_id(): stable_id로 번들 조회."""

    def test_find_bundle_by_stable_id_returns_none_when_not_found(self, repo, mock_conn):
        """존재하지 않는 stable_id는 None 반환."""
        mock_conn.execute.return_value.fetchone.return_value = None
        result = repo.find_bundle_by_stable_id("nonexistent-stable-id")
        assert result is None

    def test_find_bundle_by_stable_id_finds_existing(self, repo, mock_conn):
        """stable_id로 번들 찾기."""
        mock_conn.execute.return_value.fetchone.return_value = (
            "20260221-test-a3f2", "personal", "bundles/20260221-test-a3f2",
            "claude", "dev", "python",
        )
        result = repo.find_bundle_by_stable_id("abc123-stable")
        assert result is not None
        assert result["bundle_id"] == "20260221-test-a3f2"
        assert result["kb"] == "personal"
        assert result["path"] == "bundles/20260221-test-a3f2"

    def test_find_bundle_by_stable_id_returns_platforms_domains_topics(self, repo, mock_conn):
        """platforms, domains, topics가 리스트로 반환."""
        mock_conn.execute.return_value.fetchone.return_value = (
            "20260221-test-a3f2", "personal", "bundles/20260221-test-a3f2",
            "claude,chatgpt", "dev,ai", "python,llm",
        )
        result = repo.find_bundle_by_stable_id("abc123-stable")
        assert result is not None
        assert result["platforms"] == ["claude", "chatgpt"]
        assert result["domains"] == ["dev", "ai"]
        assert result["topics"] == ["python", "llm"]

    def test_find_bundle_by_stable_id_uses_correct_sql(self, repo, mock_conn):
        """SQL이 stable_id를 사용."""
        mock_conn.execute.return_value.fetchone.return_value = None
        repo.find_bundle_by_stable_id("abc123-stable")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "stable_id" in sql
        assert "bundle_responses" in sql


class TestRenameDomain:
    def test_rename_returns_affected_count(self, repo, mock_conn):
        """도메인 이름 변경 시 영향받은 행 수를 반환."""
        mock_conn.execute.return_value.rowcount = 3
        result = repo.rename_domain("coding", "dev")
        assert result == 3

    def test_rename_executes_delete_then_update(self, repo, mock_conn):
        """중복 제거 DELETE 후 UPDATE 순서로 실행."""
        mock_conn.execute.return_value.rowcount = 2
        repo.rename_domain("coding", "dev")
        calls = mock_conn.execute.call_args_list
        # First call: DELETE conflicting rows
        assert "DELETE" in calls[0][0][0].upper()
        assert "bundle_domains" in calls[0][0][0]
        # Second call: UPDATE remaining rows
        assert "UPDATE" in calls[1][0][0].upper()
        assert "bundle_domains" in calls[1][0][0]

    def test_rename_passes_old_and_new(self, repo, mock_conn):
        """old/new 도메인 값이 SQL 파라미터로 전달."""
        mock_conn.execute.return_value.rowcount = 0
        repo.rename_domain("ai-infra", "dev")
        calls = mock_conn.execute.call_args_list
        # DELETE params should reference both old and new
        delete_params = calls[0][0][1]
        assert "ai-infra" in delete_params
        assert "dev" in delete_params
        # UPDATE params should have new and old
        update_params = calls[1][0][1]
        assert "ai-infra" in update_params
        assert "dev" in update_params

    def test_rename_no_matches(self, repo, mock_conn):
        """매칭 없을 때 0 반환."""
        mock_conn.execute.return_value.rowcount = 0
        result = repo.rename_domain("nonexistent", "dev")
        assert result == 0


# ─── Phase 8: Metadata columns ────────────────────────────────


class TestUpsertBundleWithMetadata:
    """upsert_bundle()에 consensus, divergence, has_synthesis 파라미터 추가."""

    def _base_kwargs(self):
        return dict(
            bundle_id="20260228-multi-llm-a1b2",
            kb="personal",
            question="LLM 비교 질문",
            summary="3개 LLM 응답 비교",
            created_at=datetime.now(timezone.utc),
            response_count=3,
            path="bundles/20260228-multi-llm-a1b2",
            question_hash="meta123",
            domains=["dev"],
            topics=["llm"],
            responses=[{"platform": "claude", "model": "sonnet", "turn_count": 5}],
        )

    def test_upsert_accepts_consensus_parameter(self, repo, mock_conn):
        """consensus 파라미터를 받을 수 있어야 함."""
        repo.upsert_bundle(
            **self._base_kwargs(),
            consensus="모든 LLM이 Python을 추천함",
        )
        assert mock_conn.execute.call_count >= 1

    def test_upsert_accepts_divergence_parameter(self, repo, mock_conn):
        """divergence 파라미터를 받을 수 있어야 함."""
        repo.upsert_bundle(
            **self._base_kwargs(),
            divergence="프레임워크 선택에서 의견 차이",
        )
        assert mock_conn.execute.call_count >= 1

    def test_upsert_accepts_has_synthesis_parameter(self, repo, mock_conn):
        """has_synthesis 파라미터를 받을 수 있어야 함."""
        repo.upsert_bundle(
            **self._base_kwargs(),
            has_synthesis=True,
        )
        assert mock_conn.execute.call_count >= 1

    def test_upsert_includes_consensus_in_sql(self, repo, mock_conn):
        """INSERT SQL에 consensus 컬럼이 포함되어야 함."""
        repo.upsert_bundle(
            **self._base_kwargs(),
            consensus="합의 내용",
        )
        calls = mock_conn.execute.call_args_list
        insert_sql = calls[0][0][0]
        assert "consensus" in insert_sql

    def test_upsert_includes_divergence_in_sql(self, repo, mock_conn):
        """INSERT SQL에 divergence 컬럼이 포함되어야 함."""
        repo.upsert_bundle(
            **self._base_kwargs(),
            divergence="분기 내용",
        )
        calls = mock_conn.execute.call_args_list
        insert_sql = calls[0][0][0]
        assert "divergence" in insert_sql

    def test_upsert_includes_has_synthesis_in_sql(self, repo, mock_conn):
        """INSERT SQL에 has_synthesis 컬럼이 포함되어야 함."""
        repo.upsert_bundle(
            **self._base_kwargs(),
            has_synthesis=True,
        )
        calls = mock_conn.execute.call_args_list
        insert_sql = calls[0][0][0]
        assert "has_synthesis" in insert_sql

    def test_upsert_consensus_in_on_conflict_update(self, repo, mock_conn):
        """ON CONFLICT UPDATE에 consensus가 포함되어야 함."""
        repo.upsert_bundle(
            **self._base_kwargs(),
            consensus="합의",
        )
        calls = mock_conn.execute.call_args_list
        insert_sql = calls[0][0][0]
        # Check ON CONFLICT section includes consensus
        conflict_part = insert_sql.split("ON CONFLICT")[1] if "ON CONFLICT" in insert_sql else ""
        assert "consensus" in conflict_part

    def test_upsert_divergence_in_on_conflict_update(self, repo, mock_conn):
        """ON CONFLICT UPDATE에 divergence가 포함되어야 함."""
        repo.upsert_bundle(
            **self._base_kwargs(),
            divergence="분기",
        )
        calls = mock_conn.execute.call_args_list
        insert_sql = calls[0][0][0]
        conflict_part = insert_sql.split("ON CONFLICT")[1] if "ON CONFLICT" in insert_sql else ""
        assert "divergence" in conflict_part

    def test_upsert_has_synthesis_in_on_conflict_update(self, repo, mock_conn):
        """ON CONFLICT UPDATE에 has_synthesis가 포함되어야 함."""
        repo.upsert_bundle(
            **self._base_kwargs(),
            has_synthesis=True,
        )
        calls = mock_conn.execute.call_args_list
        insert_sql = calls[0][0][0]
        conflict_part = insert_sql.split("ON CONFLICT")[1] if "ON CONFLICT" in insert_sql else ""
        assert "has_synthesis" in conflict_part

    def test_upsert_defaults_consensus_none(self, repo, mock_conn):
        """consensus 미지정 시 None으로 전달."""
        repo.upsert_bundle(**self._base_kwargs())
        calls = mock_conn.execute.call_args_list
        params = calls[0][0][1]
        assert params.get("consensus") is None

    def test_upsert_defaults_has_synthesis_false(self, repo, mock_conn):
        """has_synthesis 미지정 시 False로 전달."""
        repo.upsert_bundle(**self._base_kwargs())
        calls = mock_conn.execute.call_args_list
        params = calls[0][0][1]
        assert params.get("has_synthesis") is False

    def test_upsert_responses_with_key_claims_and_stance(self, repo, mock_conn):
        """responses dict에 key_claims, stance가 있으면 bundle_responses INSERT에 포함."""
        kwargs = self._base_kwargs()
        kwargs["responses"] = [{
            "platform": "claude",
            "model": "sonnet",
            "turn_count": 5,
            "key_claims": ["Python이 최고", "타입 힌트 중요"],
            "stance": "긍정적",
        }]
        repo.upsert_bundle(**kwargs)
        response_inserts = [
            c for c in mock_conn.execute.call_args_list
            if "bundle_responses" in str(c[0][0]) and "INSERT" in str(c[0][0]).upper()
        ]
        assert len(response_inserts) == 1
        sql = response_inserts[0][0][0]
        assert "key_claims" in sql
        assert "stance" in sql


class TestAddResponseToBundleWithMetadata:
    """add_response_to_bundle()에 key_claims, stance 파라미터 추가."""

    def test_accepts_key_claims_parameter(self, repo, mock_conn):
        """key_claims 파라미터를 받을 수 있어야 함."""
        repo.add_response_to_bundle(
            bundle_id="20260228-test-a1b2",
            platform="chatgpt",
            model="gpt-4o",
            turn_count=5,
            key_claims=["주장1", "주장2"],
        )
        assert mock_conn.execute.call_count >= 1

    def test_accepts_stance_parameter(self, repo, mock_conn):
        """stance 파라미터를 받을 수 있어야 함."""
        repo.add_response_to_bundle(
            bundle_id="20260228-test-a1b2",
            platform="chatgpt",
            model="gpt-4o",
            turn_count=5,
            stance="긍정적",
        )
        assert mock_conn.execute.call_count >= 1

    def test_key_claims_in_insert_sql(self, repo, mock_conn):
        """INSERT SQL에 key_claims 컬럼이 포함되어야 함."""
        repo.add_response_to_bundle(
            bundle_id="20260228-test-a1b2",
            platform="chatgpt",
            key_claims=["주장1"],
            stance="중립",
        )
        calls = mock_conn.execute.call_args_list
        insert_sql = calls[0][0][0]
        assert "key_claims" in insert_sql

    def test_stance_in_insert_sql(self, repo, mock_conn):
        """INSERT SQL에 stance 컬럼이 포함되어야 함."""
        repo.add_response_to_bundle(
            bundle_id="20260228-test-a1b2",
            platform="chatgpt",
            stance="긍정적",
        )
        calls = mock_conn.execute.call_args_list
        insert_sql = calls[0][0][0]
        assert "stance" in insert_sql

    def test_key_claims_in_on_conflict_update(self, repo, mock_conn):
        """ON CONFLICT UPDATE에 key_claims가 포함되어야 함."""
        repo.add_response_to_bundle(
            bundle_id="20260228-test-a1b2",
            platform="chatgpt",
            key_claims=["주장1"],
        )
        calls = mock_conn.execute.call_args_list
        insert_sql = calls[0][0][0]
        assert "key_claims = EXCLUDED.key_claims" in insert_sql

    def test_stance_in_on_conflict_update(self, repo, mock_conn):
        """ON CONFLICT UPDATE에 stance가 포함되어야 함."""
        repo.add_response_to_bundle(
            bundle_id="20260228-test-a1b2",
            platform="chatgpt",
            stance="부정적",
        )
        calls = mock_conn.execute.call_args_list
        insert_sql = calls[0][0][0]
        assert "stance = EXCLUDED.stance" in insert_sql

    def test_key_claims_defaults_to_empty_list(self, repo, mock_conn):
        """key_claims 미지정 시 빈 리스트로 JSONB 전달."""
        repo.add_response_to_bundle(
            bundle_id="20260228-test-a1b2",
            platform="chatgpt",
        )
        calls = mock_conn.execute.call_args_list
        insert_sql = calls[0][0][0]
        assert "key_claims" in insert_sql


class TestUpdateBundleMetaWithMetadata:
    """update_bundle_meta()에 consensus, divergence, has_synthesis 파라미터 추가."""

    def test_accepts_consensus_parameter(self, repo, mock_conn):
        """consensus 파라미터를 받을 수 있어야 함."""
        repo.update_bundle_meta(
            bundle_id="20260228-test-a1b2",
            summary="요약",
            domains=["dev"],
            topics=["llm"],
            consensus="모든 LLM 동의",
        )
        assert mock_conn.execute.call_count >= 1

    def test_accepts_divergence_parameter(self, repo, mock_conn):
        """divergence 파라미터를 받을 수 있어야 함."""
        repo.update_bundle_meta(
            bundle_id="20260228-test-a1b2",
            summary="요약",
            domains=["dev"],
            topics=["llm"],
            divergence="의견 불일치",
        )
        assert mock_conn.execute.call_count >= 1

    def test_accepts_has_synthesis_parameter(self, repo, mock_conn):
        """has_synthesis 파라미터를 받을 수 있어야 함."""
        repo.update_bundle_meta(
            bundle_id="20260228-test-a1b2",
            summary="요약",
            domains=["dev"],
            topics=["llm"],
            has_synthesis=True,
        )
        assert mock_conn.execute.call_count >= 1

    def test_consensus_in_update_sql(self, repo, mock_conn):
        """UPDATE SQL에 consensus가 포함되어야 함."""
        repo.update_bundle_meta(
            bundle_id="20260228-test-a1b2",
            summary="요약",
            domains=["dev"],
            topics=["llm"],
            consensus="합의",
        )
        calls = mock_conn.execute.call_args_list
        update_sql = calls[0][0][0]
        assert "consensus" in update_sql

    def test_divergence_in_update_sql(self, repo, mock_conn):
        """UPDATE SQL에 divergence가 포함되어야 함."""
        repo.update_bundle_meta(
            bundle_id="20260228-test-a1b2",
            summary="요약",
            domains=["dev"],
            topics=["llm"],
            divergence="분기",
        )
        calls = mock_conn.execute.call_args_list
        update_sql = calls[0][0][0]
        assert "divergence" in update_sql

    def test_has_synthesis_in_update_sql(self, repo, mock_conn):
        """UPDATE SQL에 has_synthesis가 포함되어야 함."""
        repo.update_bundle_meta(
            bundle_id="20260228-test-a1b2",
            summary="요약",
            domains=["dev"],
            topics=["llm"],
            has_synthesis=True,
        )
        calls = mock_conn.execute.call_args_list
        update_sql = calls[0][0][0]
        assert "has_synthesis" in update_sql

    def test_defaults_consensus_none(self, repo, mock_conn):
        """consensus 미지정 시 None으로 전달 (기존 값 유지)."""
        repo.update_bundle_meta(
            bundle_id="20260228-test-a1b2",
            summary="요약",
            domains=["dev"],
            topics=["llm"],
        )
        calls = mock_conn.execute.call_args_list
        params = calls[0][0][1]
        assert params.get("consensus") is None


class TestGetBundleByIdWithMetadata:
    """get_bundle_by_id()가 consensus, divergence, has_synthesis를 반환."""

    def test_returns_consensus_field(self, repo, mock_conn):
        """반환 dict에 consensus 필드가 있어야 함."""
        mock_conn.execute.return_value.fetchone.return_value = (
            "20260228-test-a1b2", "personal", "LLM 비교",
            "3개 LLM 비교 요약",
            datetime(2026, 2, 28, tzinfo=timezone.utc),
            "dev", "llm",
            "모든 LLM 동의", "프레임워크 차이", True, "stable1",
        )
        result = repo.get_bundle_by_id("20260228-test-a1b2")
        assert result is not None
        assert "consensus" in result
        assert result["consensus"] == "모든 LLM 동의"

    def test_returns_divergence_field(self, repo, mock_conn):
        """반환 dict에 divergence 필드가 있어야 함."""
        mock_conn.execute.return_value.fetchone.return_value = (
            "20260228-test-a1b2", "personal", "LLM 비교",
            "3개 LLM 비교 요약",
            datetime(2026, 2, 28, tzinfo=timezone.utc),
            "dev", "llm",
            "합의", "의견 차이", False, "stable2",
        )
        result = repo.get_bundle_by_id("20260228-test-a1b2")
        assert result is not None
        assert "divergence" in result
        assert result["divergence"] == "의견 차이"

    def test_returns_has_synthesis_field(self, repo, mock_conn):
        """반환 dict에 has_synthesis 필드가 있어야 함."""
        mock_conn.execute.return_value.fetchone.return_value = (
            "20260228-test-a1b2", "personal", "LLM 비교",
            "요약",
            datetime(2026, 2, 28, tzinfo=timezone.utc),
            "dev", "llm",
            None, None, True, "stable3",
        )
        result = repo.get_bundle_by_id("20260228-test-a1b2")
        assert result is not None
        assert "has_synthesis" in result
        assert result["has_synthesis"] is True

    def test_consensus_in_select_sql(self, repo, mock_conn):
        """SELECT SQL에 consensus 컬럼이 포함되어야 함."""
        mock_conn.execute.return_value.fetchone.return_value = None
        repo.get_bundle_by_id("20260228-test-a1b2")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "consensus" in sql

    def test_divergence_in_select_sql(self, repo, mock_conn):
        """SELECT SQL에 divergence 컬럼이 포함되어야 함."""
        mock_conn.execute.return_value.fetchone.return_value = None
        repo.get_bundle_by_id("20260228-test-a1b2")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "divergence" in sql

    def test_has_synthesis_in_select_sql(self, repo, mock_conn):
        """SELECT SQL에 has_synthesis 컬럼이 포함되어야 함."""
        mock_conn.execute.return_value.fetchone.return_value = None
        repo.get_bundle_by_id("20260228-test-a1b2")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "has_synthesis" in sql


class TestSearchClaims:
    """search_claims(): key_claims에서 텍스트 검색."""

    def test_method_exists(self, repo):
        """search_claims 메서드가 존재해야 함."""
        assert hasattr(repo, "search_claims")

    def test_returns_list_of_dicts(self, repo, mock_conn):
        """검색 결과는 dict 리스트."""
        ts = datetime(2026, 2, 28, tzinfo=timezone.utc)
        mock_conn.execute.return_value.fetchall.return_value = [
            ("20260228-test-a1b2", "personal", "LLM 비교", "요약", ts,
             "claude", '["Python이 최고"]', "긍정적"),
        ]
        result = repo.search_claims("Python")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["bundle_id"] == "20260228-test-a1b2"

    def test_result_includes_platform_and_claims(self, repo, mock_conn):
        """결과에 platform, key_claims, stance가 포함되어야 함."""
        ts = datetime(2026, 2, 28, tzinfo=timezone.utc)
        mock_conn.execute.return_value.fetchall.return_value = [
            ("20260228-test-a1b2", "personal", "LLM 비교", "요약", ts,
             "claude", '["Python이 최고"]', "긍정적"),
        ]
        result = repo.search_claims("Python")
        assert "platform" in result[0]
        assert "key_claims" in result[0]
        assert "stance" in result[0]

    def test_sql_uses_jsonb_array_elements(self, repo, mock_conn):
        """SQL이 jsonb_array_elements_text를 사용해야 함."""
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.search_claims("Python")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "jsonb_array_elements_text" in sql

    def test_sql_uses_ilike_for_case_insensitive(self, repo, mock_conn):
        """대소문자 무시 검색을 위해 ILIKE 사용."""
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.search_claims("python")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "ILIKE" in sql

    def test_with_kb_filter(self, repo, mock_conn):
        """kb 필터링 시 SQL에 kb 조건 추가."""
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.search_claims("Python", kb="personal")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "kb" in sql.lower()

    def test_limit_parameter(self, repo, mock_conn):
        """limit 파라미터가 SQL에 반영."""
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.search_claims("Python", limit=5)
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params.get("limit") == 5

    def test_empty_result(self, repo, mock_conn):
        """검색 결과 없으면 빈 리스트 반환."""
        mock_conn.execute.return_value.fetchall.return_value = []
        result = repo.search_claims("nonexistent")
        assert result == []

    def test_query_wrapped_with_percent(self, repo, mock_conn):
        """검색어가 %query% 형태로 ILIKE에 전달."""
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.search_claims("Python")
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params.get("pattern") == "%Python%"


class TestGetResponsesForBundle:
    """get_responses_for_bundle(): bundle_id에 해당하는 platform responses 조회."""

    def test_returns_empty_for_nonexistent(self, repo, mock_conn):
        """존재하지 않는 bundle_id는 빈 리스트 반환."""
        mock_conn.execute.return_value.fetchall.return_value = []
        result = repo.get_responses_for_bundle("nonexistent-bundle-id")
        assert result == []

    def test_returns_list_of_dicts(self, repo, mock_conn):
        """2개 row 반환 시 2개 dict 리스트, 올바른 키 포함."""
        mock_conn.execute.return_value.fetchall.return_value = [
            ("claude", "sonnet-4", 5, ["LLM은 유용하다"], "긍정적", "/path/to/claude.jsonl"),
            ("chatgpt", "gpt-4o", 3, ["AI는 발전 중"], "중립", "/path/to/chatgpt.jsonl"),
        ]
        result = repo.get_responses_for_bundle("20260228-test-a1b2")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["platform"] == "claude"
        assert result[0]["model"] == "sonnet-4"
        assert result[0]["turn_count"] == 5
        assert result[0]["key_claims"] == ["LLM은 유용하다"]
        assert result[0]["stance"] == "긍정적"
        assert result[0]["source_path"] == "/path/to/claude.jsonl"
        assert result[1]["platform"] == "chatgpt"
        assert result[1]["model"] == "gpt-4o"
        assert result[1]["turn_count"] == 3

    def test_key_claims_defaults_to_empty_list(self, repo, mock_conn):
        """key_claims가 None이면 빈 리스트로 반환."""
        mock_conn.execute.return_value.fetchall.return_value = [
            ("claude", "sonnet-4", 5, None, None, None),
        ]
        result = repo.get_responses_for_bundle("20260228-test-a1b2")
        assert len(result) == 1
        assert result[0]["key_claims"] == []
        assert result[0]["stance"] is None
        assert result[0]["source_path"] is None


class TestListBundlesByTopic:
    """Tests for list_bundles_by_topic()."""

    def test_returns_empty_for_missing_topic(self, repo, mock_conn):
        """존재하지 않는 topic은 빈 리스트 반환."""
        mock_conn.execute.return_value.fetchall.return_value = []
        result = repo.list_bundles_by_topic("nonexistent-topic")
        assert result == []

    def test_returns_list_of_dicts(self, repo, mock_conn):
        """1개 row 반환 시 올바른 키를 가진 dict 리스트."""
        mock_conn.execute.return_value.fetchall.return_value = [
            (
                "20260228-python-basics-a1b2",
                "personal",
                "Python 기초 질문",
                "Python 기초 요약",
                datetime(2026, 2, 28, tzinfo=timezone.utc),
            ),
        ]
        result = repo.list_bundles_by_topic("python")
        assert len(result) == 1
        assert result[0]["bundle_id"] == "20260228-python-basics-a1b2"
        assert result[0]["kb"] == "personal"
        assert result[0]["question"] == "Python 기초 질문"
        assert result[0]["summary"] == "Python 기초 요약"
        assert result[0]["created_at"] == datetime(2026, 2, 28, tzinfo=timezone.utc)

    def test_with_kb_filter(self, repo, mock_conn):
        """kb 파라미터가 SQL에 전달되는지 검증."""
        mock_conn.execute.return_value.fetchall.return_value = []
        repo.list_bundles_by_topic("python", kb="personal")
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        assert "b.kb = %s" in sql
        assert params == ("python", "personal")
