"""Tests for ChunkStore with embedder injection (client-side embedding)."""

from unittest.mock import MagicMock, patch

import pytest

from pkb.db.chromadb_client import ChunkStore
from pkb.embedding.server_side import ServerSideEmbedder
from pkb.embedding.tei_embedder import TEIEmbedder
from pkb.models.config import ChromaDBConfig


@pytest.fixture
def mock_chroma():
    client = MagicMock()
    collection = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client, collection


@pytest.fixture
def tei_embedder():
    e = MagicMock(spec=TEIEmbedder)
    e.model_name = "BAAI/bge-m3"
    e.dimensions = 1024
    return e


class TestChunkStoreWithEmbedder:
    """embedder 주입 시 client-side 임베딩 동작."""

    def test_init_with_embedder_sets_collection_metadata(self, mock_chroma, tei_embedder):
        client, _ = mock_chroma
        config = ChromaDBConfig(host="localhost", port=8000)
        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            ChunkStore(config, embedder=tei_embedder)
        call_kwargs = client.get_or_create_collection.call_args
        metadata = call_kwargs.kwargs.get("metadata", {})
        assert metadata.get("embedding_model") == "BAAI/bge-m3"
        assert metadata.get("embedding_dimensions") == 1024
        assert metadata.get("hnsw:space") == "cosine"

    def test_init_without_embedder_no_metadata(self, mock_chroma):
        client, _ = mock_chroma
        config = ChromaDBConfig(host="localhost", port=8000)
        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            ChunkStore(config)
        call_kwargs = client.get_or_create_collection.call_args
        # No metadata arg, or metadata without embedding_model
        metadata = call_kwargs.kwargs.get("metadata")
        assert metadata is None

    def test_upsert_with_embedder_sends_embeddings(self, mock_chroma, tei_embedder):
        client, collection = mock_chroma
        config = ChromaDBConfig()
        tei_embedder.embed_documents.return_value = [[0.1] * 1024, [0.2] * 1024]

        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            store = ChunkStore(config, embedder=tei_embedder)

        chunks = [
            {"id": "c1", "document": "hello", "metadata": {"bundle_id": "b1"}},
            {"id": "c2", "document": "world", "metadata": {"bundle_id": "b1"}},
        ]
        store.upsert_chunks(chunks)

        tei_embedder.embed_documents.assert_called_once_with(["hello", "world"])
        upsert_kwargs = collection.upsert.call_args.kwargs
        assert "embeddings" in upsert_kwargs
        assert len(upsert_kwargs["embeddings"]) == 2

    def test_upsert_without_embedder_sends_documents(self, mock_chroma):
        """기존 동작: embedder 없으면 documents만 전송."""
        client, collection = mock_chroma
        config = ChromaDBConfig()

        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            store = ChunkStore(config)

        chunks = [{"id": "c1", "document": "hello", "metadata": {"bundle_id": "b1"}}]
        store.upsert_chunks(chunks)

        upsert_kwargs = collection.upsert.call_args.kwargs
        assert "documents" in upsert_kwargs
        assert "embeddings" not in upsert_kwargs

    def test_search_with_embedder_uses_query_embeddings(self, mock_chroma, tei_embedder):
        client, collection = mock_chroma
        config = ChromaDBConfig()
        tei_embedder.embed_query.return_value = [0.5] * 1024
        collection.query.return_value = {
            "ids": [["c1"]],
            "documents": [["doc1"]],
            "metadatas": [[{"bundle_id": "b1"}]],
            "distances": [[0.1]],
        }

        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            store = ChunkStore(config, embedder=tei_embedder)

        results = store.search("query text", n_results=5)
        tei_embedder.embed_query.assert_called_once_with("query text")
        query_kwargs = collection.query.call_args.kwargs
        assert "query_embeddings" in query_kwargs
        assert "query_texts" not in query_kwargs
        assert len(results) == 1

    def test_search_without_embedder_uses_query_texts(self, mock_chroma):
        """기존 동작: embedder 없으면 query_texts 사용."""
        client, collection = mock_chroma
        config = ChromaDBConfig()
        collection.query.return_value = {
            "ids": [["c1"]],
            "documents": [["doc1"]],
            "metadatas": [[{"bundle_id": "b1"}]],
            "distances": [[0.1]],
        }

        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            store = ChunkStore(config)

        store.search("query text")
        query_kwargs = collection.query.call_args.kwargs
        assert "query_texts" in query_kwargs
        assert "query_embeddings" not in query_kwargs

    def test_server_side_embedder_treated_as_no_embedder(self, mock_chroma):
        """ServerSideEmbedder를 주입해도 server-side 동작 유지."""
        client, collection = mock_chroma
        config = ChromaDBConfig()
        sentinel = ServerSideEmbedder()

        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            store = ChunkStore(config, embedder=sentinel)

        chunks = [{"id": "c1", "document": "hello", "metadata": {"bundle_id": "b1"}}]
        store.upsert_chunks(chunks)

        upsert_kwargs = collection.upsert.call_args.kwargs
        assert "documents" in upsert_kwargs
        assert "embeddings" not in upsert_kwargs


class TestChunkStoreCollectionInfo:
    """get_collection_model_info 메서드 테스트."""

    def test_get_collection_model_info_with_metadata(self, mock_chroma):
        client, collection = mock_chroma
        collection.metadata = {
            "embedding_model": "BAAI/bge-m3",
            "embedding_dimensions": 1024,
        }
        config = ChromaDBConfig()
        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            store = ChunkStore(config)
        info = store.get_collection_model_info()
        assert info["embedding_model"] == "BAAI/bge-m3"
        assert info["embedding_dimensions"] == 1024

    def test_get_collection_model_info_no_metadata(self, mock_chroma):
        client, collection = mock_chroma
        collection.metadata = {}
        config = ChromaDBConfig()
        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            store = ChunkStore(config)
        info = store.get_collection_model_info()
        assert info.get("embedding_model") is None

    def test_get_collection_model_info_none_metadata(self, mock_chroma):
        client, collection = mock_chroma
        collection.metadata = None
        config = ChromaDBConfig()
        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            store = ChunkStore(config)
        info = store.get_collection_model_info()
        assert info == {}


class TestChunkStoreRecreateCollection:
    """drop_and_recreate_collection 메서드 테스트."""

    def test_drop_and_recreate(self, mock_chroma, tei_embedder):
        client, collection = mock_chroma
        new_collection = MagicMock()
        client.get_or_create_collection.return_value = new_collection
        config = ChromaDBConfig()

        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            store = ChunkStore(config, embedder=tei_embedder)

        store.drop_and_recreate_collection()
        client.delete_collection.assert_called_once_with(name=config.collection)
