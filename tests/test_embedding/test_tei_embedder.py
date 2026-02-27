"""Tests for TEIEmbedder — wraps TEIClient with Embedder interface."""

from unittest.mock import MagicMock

import pytest

from pkb.embedding.tei_embedder import TEIEmbedder


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def embedder(mock_client):
    return TEIEmbedder(
        client=mock_client,
        model_name="BAAI/bge-m3",
        dimensions=1024,
        batch_size=2,
    )


class TestTEIEmbedder:
    def test_is_embedder(self, embedder):
        from pkb.embedding.base import Embedder
        assert isinstance(embedder, Embedder)

    def test_model_name(self, embedder):
        assert embedder.model_name == "BAAI/bge-m3"

    def test_dimensions(self, embedder):
        assert embedder.dimensions == 1024

    def test_embed_documents_single_batch(self, embedder, mock_client):
        mock_client.embed.return_value = [[0.1] * 1024]
        result = embedder.embed_documents(["text1"])
        assert len(result) == 1
        mock_client.embed.assert_called_once_with(["text1"])

    def test_embed_documents_multi_batch(self, embedder, mock_client):
        """batch_size=2이므로 3개 텍스트는 2번 호출."""
        mock_client.embed.side_effect = [
            [[0.1] * 1024, [0.2] * 1024],
            [[0.3] * 1024],
        ]
        result = embedder.embed_documents(["a", "b", "c"])
        assert len(result) == 3
        assert mock_client.embed.call_count == 2

    def test_embed_documents_empty(self, embedder, mock_client):
        result = embedder.embed_documents([])
        assert result == []
        mock_client.embed.assert_not_called()

    def test_embed_query(self, embedder, mock_client):
        mock_client.embed.return_value = [[0.5] * 1024]
        result = embedder.embed_query("query text")
        assert len(result) == 1024
        mock_client.embed.assert_called_once_with(["query text"])
