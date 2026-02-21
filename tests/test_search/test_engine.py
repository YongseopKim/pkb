"""Tests for SearchEngine (mock-based)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from pkb.db.chromadb_client import SearchResult
from pkb.search.engine import SearchEngine
from pkb.search.models import SearchMode, SearchQuery


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def mock_chunk_store():
    return MagicMock()


@pytest.fixture
def engine(mock_repo, mock_chunk_store):
    return SearchEngine(repo=mock_repo, chunk_store=mock_chunk_store)


class TestSearchEngineConstruction:
    def test_construction(self, engine):
        assert engine is not None


class TestKeywordSearch:
    def test_keyword_search_returns_results(self, engine, mock_repo):
        mock_repo.search_fts.return_value = [
            {
                "bundle_id": "20260221-bitcoin-a3f2",
                "kb": "personal",
                "question": "Bitcoin halving?",
                "summary": "비트코인 반감기 분석",
                "created_at": datetime(2026, 2, 21, tzinfo=timezone.utc),
                "domains": "investing",
                "topics": "bitcoin,crypto",
                "rank": 0.5,
            },
        ]
        query = SearchQuery(query="bitcoin", mode=SearchMode.KEYWORD)
        results = engine.search(query)

        assert len(results) == 1
        assert results[0].bundle_id == "20260221-bitcoin-a3f2"
        assert results[0].source == "fts"
        assert 0.0 <= results[0].score <= 1.0

    def test_keyword_search_empty(self, engine, mock_repo):
        mock_repo.search_fts.return_value = []
        query = SearchQuery(query="nonexistent", mode=SearchMode.KEYWORD)
        results = engine.search(query)
        assert results == []

    def test_keyword_search_normalizes_scores(self, engine, mock_repo):
        """Multiple results should have scores normalized to 0-1."""
        mock_repo.search_fts.return_value = [
            {
                "bundle_id": "b1", "kb": "p", "question": "q1", "summary": "s1",
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "domains": "d1", "topics": "t1", "rank": 1.0,
            },
            {
                "bundle_id": "b2", "kb": "p", "question": "q2", "summary": "s2",
                "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
                "domains": "d2", "topics": "t2", "rank": 0.5,
            },
            {
                "bundle_id": "b3", "kb": "p", "question": "q3", "summary": "s3",
                "created_at": datetime(2026, 1, 3, tzinfo=timezone.utc),
                "domains": "d3", "topics": "t3", "rank": 0.1,
            },
        ]
        query = SearchQuery(query="test", mode=SearchMode.KEYWORD, limit=10)
        results = engine.search(query)
        assert len(results) == 3
        # Highest rank should get score 1.0
        assert results[0].score == 1.0
        # All scores should be in [0, 1]
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_keyword_passes_filters(self, engine, mock_repo):
        """Filters should be passed through to repo."""
        mock_repo.search_fts.return_value = []
        from datetime import date
        query = SearchQuery(
            query="test", mode=SearchMode.KEYWORD,
            domains=["investing"], topics=["bitcoin"],
            kb="personal", after=date(2026, 1, 1), before=date(2026, 12, 31),
            limit=5,
        )
        engine.search(query)
        mock_repo.search_fts.assert_called_once_with(
            query="test", kb="personal", domains=["investing"], topics=["bitcoin"],
            after=date(2026, 1, 1), before=date(2026, 12, 31), limit=5,
        )


class TestSemanticSearch:
    def test_semantic_search_returns_results(self, engine, mock_chunk_store):
        mock_chunk_store.search.return_value = [
            SearchResult(
                chunk_id="b1-chunk-0", document="doc text",
                metadata={"bundle_id": "b1", "kb": "personal",
                          "domains": "investing", "topics": "bitcoin"},
                distance=0.2,
            ),
            SearchResult(
                chunk_id="b1-chunk-1", document="another chunk",
                metadata={"bundle_id": "b1", "kb": "personal",
                          "domains": "investing", "topics": "bitcoin"},
                distance=0.3,
            ),
            SearchResult(
                chunk_id="b2-chunk-0", document="different bundle",
                metadata={"bundle_id": "b2", "kb": "personal",
                          "domains": "dev", "topics": "python"},
                distance=0.5,
            ),
        ]
        engine._repo.get_bundle_by_id.side_effect = [
            {
                "bundle_id": "b1", "kb": "personal", "question": "Q1",
                "summary": "S1", "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "domains": "investing", "topics": "bitcoin",
            },
            {
                "bundle_id": "b2", "kb": "personal", "question": "Q2",
                "summary": "S2", "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
                "domains": "dev", "topics": "python",
            },
        ]
        query = SearchQuery(query="bitcoin", mode=SearchMode.SEMANTIC)
        results = engine.search(query)

        # Two bundles (chunks aggregated per bundle)
        assert len(results) == 2
        assert results[0].source == "semantic"
        # b1 has closer distance so should rank higher
        assert results[0].bundle_id == "b1"
        assert results[0].score >= results[1].score

    def test_semantic_search_empty(self, engine, mock_chunk_store):
        mock_chunk_store.search.return_value = []
        query = SearchQuery(query="nonexistent", mode=SearchMode.SEMANTIC)
        results = engine.search(query)
        assert results == []

    def test_semantic_with_kb_filter(self, engine, mock_chunk_store):
        """KB filter should be passed as where clause."""
        mock_chunk_store.search.return_value = []
        query = SearchQuery(query="test", mode=SearchMode.SEMANTIC, kb="personal")
        engine.search(query)
        mock_chunk_store.search.assert_called_once()
        call_kwargs = mock_chunk_store.search.call_args
        assert call_kwargs.kwargs.get("where") == {"kb": "personal"}


class TestHybridSearch:
    def test_hybrid_merges_both_sources(self, engine, mock_repo, mock_chunk_store):
        """Hybrid search combines FTS and semantic results."""
        # FTS returns b1, b2
        mock_repo.search_fts.return_value = [
            {
                "bundle_id": "b1", "kb": "p", "question": "Q1", "summary": "S1",
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "domains": "d1", "topics": "t1", "rank": 1.0,
            },
            {
                "bundle_id": "b2", "kb": "p", "question": "Q2", "summary": "S2",
                "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
                "domains": "d2", "topics": "t2", "rank": 0.5,
            },
        ]
        # Semantic returns b1, b3
        mock_chunk_store.search.return_value = [
            SearchResult(
                chunk_id="b1-chunk-0", document="doc",
                metadata={"bundle_id": "b1", "kb": "p",
                          "domains": "d1", "topics": "t1"},
                distance=0.1,
            ),
            SearchResult(
                chunk_id="b3-chunk-0", document="doc",
                metadata={"bundle_id": "b3", "kb": "p",
                          "domains": "d3", "topics": "t3"},
                distance=0.3,
            ),
        ]
        mock_repo.get_bundle_by_id.side_effect = [
            {
                "bundle_id": "b1", "kb": "p", "question": "Q1", "summary": "S1",
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "domains": "d1", "topics": "t1",
            },
            {
                "bundle_id": "b3", "kb": "p", "question": "Q3", "summary": "S3",
                "created_at": datetime(2026, 1, 3, tzinfo=timezone.utc),
                "domains": "d3", "topics": "t3",
            },
        ]

        query = SearchQuery(query="test", mode=SearchMode.HYBRID)
        results = engine.search(query)

        # Should have 3 unique bundles: b1 (both), b2 (fts-only), b3 (semantic-only)
        bundle_ids = {r.bundle_id for r in results}
        assert bundle_ids == {"b1", "b2", "b3"}

        # b1 should be "both"
        b1 = next(r for r in results if r.bundle_id == "b1")
        assert b1.source == "both"

        # b2 should be "fts" only
        b2 = next(r for r in results if r.bundle_id == "b2")
        assert b2.source == "fts"

        # b3 should be "semantic" only
        b3 = next(r for r in results if r.bundle_id == "b3")
        assert b3.source == "semantic"

    def test_hybrid_both_source_scores_higher(self, engine, mock_repo, mock_chunk_store):
        """Bundle found in both sources should score higher than single-source."""
        # Both return b1; only FTS returns b2
        mock_repo.search_fts.return_value = [
            {
                "bundle_id": "b1", "kb": "p", "question": "Q1", "summary": "S1",
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "domains": "d", "topics": "t", "rank": 0.8,
            },
            {
                "bundle_id": "b2", "kb": "p", "question": "Q2", "summary": "S2",
                "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
                "domains": "d", "topics": "t", "rank": 0.8,
            },
        ]
        mock_chunk_store.search.return_value = [
            SearchResult(
                chunk_id="b1-chunk-0", document="doc",
                metadata={"bundle_id": "b1", "kb": "p", "domains": "d", "topics": "t"},
                distance=0.2,
            ),
        ]
        mock_repo.get_bundle_by_id.return_value = {
            "bundle_id": "b1", "kb": "p", "question": "Q1", "summary": "S1",
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "domains": "d", "topics": "t",
        }

        query = SearchQuery(query="test", mode=SearchMode.HYBRID)
        results = engine.search(query)

        b1 = next(r for r in results if r.bundle_id == "b1")
        b2 = next(r for r in results if r.bundle_id == "b2")
        # b1 (both sources) should beat b2 (fts-only, same rank)
        assert b1.score > b2.score

    def test_hybrid_respects_limit(self, engine, mock_repo, mock_chunk_store):
        mock_repo.search_fts.return_value = []
        mock_chunk_store.search.return_value = []
        query = SearchQuery(query="test", mode=SearchMode.HYBRID, limit=3)
        results = engine.search(query)
        assert len(results) <= 3

    def test_hybrid_default_mode(self, engine, mock_repo, mock_chunk_store):
        """Default search mode should be hybrid."""
        mock_repo.search_fts.return_value = []
        mock_chunk_store.search.return_value = []
        query = SearchQuery(query="test")
        engine.search(query)
        # Both FTS and semantic should be called
        mock_repo.search_fts.assert_called_once()
        mock_chunk_store.search.assert_called_once()
