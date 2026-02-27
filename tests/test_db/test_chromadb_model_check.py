"""Tests for ChunkStore model consistency check."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from pkb.db.chromadb_client import ChunkStore
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


class TestModelConsistencyCheck:
    """_check_model_consistency 동작 검증."""

    def test_matching_model_no_warning(self, mock_chroma, tei_embedder, caplog):
        """모델이 일치하면 경고 없음."""
        client, collection = mock_chroma
        collection.metadata = {
            "embedding_model": "BAAI/bge-m3",
            "embedding_dimensions": 1024,
        }
        config = ChromaDBConfig()
        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            with caplog.at_level(logging.WARNING):
                ChunkStore(config, embedder=tei_embedder)
        assert "model mismatch" not in caplog.text.lower()
        assert "reembed" not in caplog.text.lower()

    def test_mismatched_model_warns(self, mock_chroma, tei_embedder, caplog):
        """collection 모델 ≠ config 모델 → WARNING."""
        client, collection = mock_chroma
        collection.metadata = {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimensions": 384,
        }
        config = ChromaDBConfig()
        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            with caplog.at_level(logging.WARNING):
                ChunkStore(config, embedder=tei_embedder)
        assert "mismatch" in caplog.text.lower() or "reembed" in caplog.text.lower()

    def test_legacy_no_metadata_warns(self, mock_chroma, tei_embedder, caplog):
        """collection에 metadata가 없음 (legacy) → WARNING."""
        client, collection = mock_chroma
        collection.metadata = {}
        config = ChromaDBConfig()
        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            with caplog.at_level(logging.WARNING):
                ChunkStore(config, embedder=tei_embedder)
        assert "legacy" in caplog.text.lower() or "no embedding" in caplog.text.lower()

    def test_none_metadata_warns(self, mock_chroma, tei_embedder, caplog):
        """collection.metadata가 None (legacy) → WARNING."""
        client, collection = mock_chroma
        collection.metadata = None
        config = ChromaDBConfig()
        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            with caplog.at_level(logging.WARNING):
                ChunkStore(config, embedder=tei_embedder)
        assert "legacy" in caplog.text.lower() or "no embedding" in caplog.text.lower()

    def test_no_embedder_skips_check(self, mock_chroma, caplog):
        """embedder 없으면 (server-side) 검사 건너뜀."""
        client, collection = mock_chroma
        collection.metadata = {}
        config = ChromaDBConfig()
        with patch("pkb.db.chromadb_client.chromadb.HttpClient", return_value=client):
            with caplog.at_level(logging.WARNING):
                ChunkStore(config)
        assert "mismatch" not in caplog.text.lower()
        assert "reembed" not in caplog.text.lower()
