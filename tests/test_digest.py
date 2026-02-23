"""Tests for DigestEngine — topic/domain knowledge summaries."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from pkb.models.config import DigestConfig


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def mock_search_engine():
    return MagicMock()


@pytest.fixture
def mock_router():
    return MagicMock()


@pytest.fixture
def engine(mock_repo, mock_search_engine, mock_router):
    config = DigestConfig()
    from pkb.digest import DigestEngine

    return DigestEngine(
        repo=mock_repo,
        search_engine=mock_search_engine,
        router=mock_router,
        config=config,
    )


class TestDigestByTopic:
    def test_generates_digest_for_topic(self, engine, mock_search_engine, mock_router):
        """digest_topic should gather bundles and produce an LLM summary."""
        from pkb.search.models import BundleSearchResult

        mock_search_engine.search.return_value = [
            BundleSearchResult(
                bundle_id="20260101-test-abc1",
                question="Python async는?",
                summary="async/await 패턴 설명",
                domains=["dev"],
                topics=["python"],
                score=0.9,
                created_at=datetime(2026, 1, 1),
                source="both",
            ),
        ]
        mock_router.complete.return_value = "Python async에 대한 종합 요약입니다."

        result = engine.digest_topic("python", kb="personal")
        assert result.content is not None
        assert len(result.sources) >= 1
        mock_router.complete.assert_called_once()

    def test_empty_topic_returns_no_data(self, engine, mock_search_engine):
        """Topic with no bundles should return appropriate message."""
        mock_search_engine.search.return_value = []

        result = engine.digest_topic("nonexistent")
        assert "없" in result.content or "찾을 수 없" in result.content


class TestDigestByDomain:
    def test_generates_digest_for_domain(self, engine, mock_repo, mock_router):
        """digest_domain should gather domain bundles and summarize."""
        mock_repo.list_bundles_by_domain.return_value = [
            {
                "bundle_id": "20260101-test-abc1",
                "question": "투자 전략은?",
                "summary": "투자 전략 설명",
                "kb": "personal",
            },
        ]
        mock_router.complete.return_value = "투자 도메인 종합 요약입니다."

        result = engine.digest_domain("투자", kb="personal")
        assert result.content is not None
        assert len(result.sources) >= 1

    def test_empty_domain_returns_no_data(self, engine, mock_repo):
        """Domain with no bundles should return appropriate message."""
        mock_repo.list_bundles_by_domain.return_value = []

        result = engine.digest_domain("nonexistent")
        assert "없" in result.content or "찾을 수 없" in result.content


class TestDigestReport:
    def test_report_includes_bundle_count(self, engine, mock_search_engine, mock_router):
        """Report should include metadata about bundles used."""
        from pkb.search.models import BundleSearchResult

        mock_search_engine.search.return_value = [
            BundleSearchResult(
                bundle_id=f"20260101-test-{i:04d}",
                question=f"Q{i}",
                summary=f"S{i}",
                domains=["dev"],
                topics=["python"],
                score=0.8,
                created_at=datetime(2026, 1, 1),
                source="both",
            )
            for i in range(3)
        ]
        mock_router.complete.return_value = "종합 요약"

        result = engine.digest_topic("python")
        assert result.bundle_count == 3

    def test_digest_result_has_topic_field(self, engine, mock_search_engine, mock_router):
        """DigestResult should store what was digested."""
        mock_search_engine.search.return_value = []
        result = engine.digest_topic("python")
        assert result.topic == "python"

    def test_digest_result_has_domain_field(self, engine, mock_repo):
        """DigestResult should store domain when digesting by domain."""
        mock_repo.list_bundles_by_domain.return_value = []
        result = engine.digest_domain("dev")
        assert result.domain == "dev"
