"""Tests for ChromaDB client (mock-based)."""

from unittest.mock import MagicMock, patch

import pytest

from pkb.db.chromadb_client import ChunkStore
from pkb.models.config import ChromaDBConfig


@pytest.fixture
def mock_chroma_client():
    client = MagicMock()
    collection = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client, collection


@pytest.fixture
def store(mock_chroma_client):
    client, collection = mock_chroma_client
    config = ChromaDBConfig(host="localhost", port=8000)
    with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
        s = ChunkStore(config)
    return s


class TestChunkStore:
    def test_construction(self, store):
        assert store is not None

    def test_upsert_chunks(self, store, mock_chroma_client):
        _, collection = mock_chroma_client
        chunks = [
            {
                "id": "bundle1-chunk-0",
                "document": "청크 텍스트 내용",
                "metadata": {
                    "bundle_id": "20260221-test-a3f2",
                    "kb": "personal",
                    "platform": "claude",
                    "domains": "dev",
                    "topics": "python,async",
                },
            },
            {
                "id": "bundle1-chunk-1",
                "document": "두 번째 청크",
                "metadata": {
                    "bundle_id": "20260221-test-a3f2",
                    "kb": "personal",
                    "platform": "claude",
                    "domains": "dev",
                    "topics": "python",
                },
            },
        ]
        store.upsert_chunks(chunks)
        collection.upsert.assert_called_once()
        call_kwargs = collection.upsert.call_args
        assert len(call_kwargs.kwargs["ids"]) == 2

    def test_delete_by_bundle(self, store, mock_chroma_client):
        _, collection = mock_chroma_client
        store.delete_by_bundle("20260221-test-a3f2")
        collection.delete.assert_called_once()

    def test_search(self, store, mock_chroma_client):
        _, collection = mock_chroma_client
        collection.query.return_value = {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"bundle_id": "b1"}, {"bundle_id": "b2"}]],
            "distances": [[0.1, 0.2]],
        }
        results = store.search("검색 쿼리", n_results=2)
        collection.query.assert_called_once()
        assert len(results) == 2

    def test_search_with_where_filter(self, store, mock_chroma_client):
        _, collection = mock_chroma_client
        collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["doc1"]],
            "metadatas": [[{"bundle_id": "b1", "kb": "personal"}]],
            "distances": [[0.1]],
        }
        results = store.search("query", n_results=5, where={"kb": "personal"})
        assert len(results) == 1
        # Verify where was passed to ChromaDB
        call_kwargs = collection.query.call_args
        assert call_kwargs.kwargs.get("where") == {"kb": "personal"}

    def test_search_without_where(self, store, mock_chroma_client):
        """Search without where filter should not pass where to query."""
        _, collection = mock_chroma_client
        collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["doc1"]],
            "metadatas": [[{"bundle_id": "b1"}]],
            "distances": [[0.1]],
        }
        results = store.search("query", n_results=5)
        assert len(results) == 1
        call_kwargs = collection.query.call_args
        assert call_kwargs.kwargs.get("where") is None

    def test_delete_by_kb(self, store, mock_chroma_client):
        """KB별 청크 삭제 — where 파라미터 검증."""
        _, collection = mock_chroma_client
        store.delete_by_kb("personal")
        collection.delete.assert_called_once_with(where={"kb": "personal"})

    def test_heartbeat(self, store, mock_chroma_client):
        client, _ = mock_chroma_client
        client.heartbeat.return_value = 1234567890
        result = store.heartbeat()
        assert result == 1234567890
