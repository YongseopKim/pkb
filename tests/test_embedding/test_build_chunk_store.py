"""Tests for build_chunk_store helper."""

from unittest.mock import MagicMock, patch

from pkb.config import build_chunk_store
from pkb.db.chromadb_client import ChunkStore
from pkb.models.config import EmbeddingConfig, PKBConfig


class TestBuildChunkStore:
    def test_server_mode_returns_chunk_store(self):
        config = PKBConfig(embedding=EmbeddingConfig(mode="server"))
        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=MagicMock()):
            store = build_chunk_store(config)
        assert isinstance(store, ChunkStore)

    def test_server_mode_no_client_side(self):
        config = PKBConfig(embedding=EmbeddingConfig(mode="server"))
        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=MagicMock()):
            store = build_chunk_store(config)
        assert not store._use_client_side

    def test_tei_mode_returns_chunk_store_with_embedder(self):
        config = PKBConfig(embedding=EmbeddingConfig(
            mode="tei",
            model_name="BAAI/bge-m3",
            dimensions=1024,
            tei_url="http://localhost:8090",
        ))
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.metadata = {
            "embedding_model": "BAAI/bge-m3",
            "embedding_dimensions": 1024,
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=mock_client):
            store = build_chunk_store(config)
        assert isinstance(store, ChunkStore)
        assert store._use_client_side is True

    def test_default_config_returns_tei_mode(self):
        """Default config is now TEI (bge-m3)."""
        config = PKBConfig()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.metadata = {
            "embedding_model": "BAAI/bge-m3",
            "embedding_dimensions": 1024,
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=mock_client):
            store = build_chunk_store(config)
        assert store._use_client_side is True
