"""Tests for search models."""

from datetime import date, datetime, timezone

import pytest

from pkb.search.models import BundleSearchResult, SearchMode, SearchQuery


class TestSearchMode:
    def test_enum_values(self):
        assert SearchMode.KEYWORD == "keyword"
        assert SearchMode.SEMANTIC == "semantic"
        assert SearchMode.HYBRID == "hybrid"

    def test_all_modes(self):
        assert len(SearchMode) == 3


class TestSearchQuery:
    def test_minimal_query(self):
        q = SearchQuery(query="bitcoin halving")
        assert q.query == "bitcoin halving"
        assert q.mode == SearchMode.HYBRID  # default
        assert q.domains == []
        assert q.topics == []
        assert q.kb is None
        assert q.after is None
        assert q.before is None
        assert q.limit == 10

    def test_full_query(self):
        q = SearchQuery(
            query="bitcoin halving",
            mode=SearchMode.KEYWORD,
            domains=["investing"],
            topics=["bitcoin", "crypto"],
            kb="personal",
            after=date(2026, 1, 1),
            before=date(2026, 12, 31),
            limit=5,
        )
        assert q.mode == SearchMode.KEYWORD
        assert q.domains == ["investing"]
        assert q.kb == "personal"
        assert q.limit == 5

    def test_query_required(self):
        with pytest.raises(Exception):
            SearchQuery()  # type: ignore[call-arg]

    def test_empty_query_rejected(self):
        with pytest.raises(Exception):
            SearchQuery(query="")

    def test_limit_positive(self):
        with pytest.raises(Exception):
            SearchQuery(query="test", limit=0)

    def test_search_query_accepts_stance_filter(self):
        """SearchQuery should accept stance filter field."""
        q = SearchQuery(query="test", stance="informative")
        assert q.stance == "informative"

    def test_search_query_accepts_has_consensus_filter(self):
        """SearchQuery should accept has_consensus filter field."""
        q = SearchQuery(query="test", has_consensus=True)
        assert q.has_consensus is True

    def test_search_query_accepts_has_synthesis_filter(self):
        """SearchQuery should accept has_synthesis filter field."""
        q = SearchQuery(query="test", has_synthesis=True)
        assert q.has_synthesis is True

    def test_search_query_new_filters_default_none(self):
        """New filter fields should default to None."""
        q = SearchQuery(query="test")
        assert q.stance is None
        assert q.has_consensus is None
        assert q.has_synthesis is None


class TestBundleSearchResult:
    def test_construction(self):
        r = BundleSearchResult(
            bundle_id="20260221-bitcoin-a3f2",
            question="Bitcoin halving 이후 가격 전망은?",
            summary="비트코인 반감기 분석",
            domains=["investing"],
            topics=["bitcoin", "crypto"],
            score=0.87,
            created_at=datetime(2026, 2, 21, tzinfo=timezone.utc),
            source="both",
        )
        assert r.bundle_id == "20260221-bitcoin-a3f2"
        assert r.score == 0.87
        assert r.source == "both"

    def test_source_values(self):
        """source must be one of fts, semantic, both."""
        for src in ("fts", "semantic", "both"):
            r = BundleSearchResult(
                bundle_id="test",
                question="q",
                summary="s",
                domains=[],
                topics=[],
                score=0.5,
                created_at=datetime.now(timezone.utc),
                source=src,
            )
            assert r.source == src

    def test_invalid_source_rejected(self):
        with pytest.raises(Exception):
            BundleSearchResult(
                bundle_id="test",
                question="q",
                summary="s",
                domains=[],
                topics=[],
                score=0.5,
                created_at=datetime.now(timezone.utc),
                source="invalid",
            )

    def test_score_range(self):
        """Score should be between 0 and 1."""
        with pytest.raises(Exception):
            BundleSearchResult(
                bundle_id="test",
                question="q",
                summary="s",
                domains=[],
                topics=[],
                score=1.5,
                created_at=datetime.now(timezone.utc),
                source="fts",
            )
