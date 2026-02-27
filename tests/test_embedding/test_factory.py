"""Tests for create_embedder factory."""

import pytest

from pkb.embedding.factory import create_embedder
from pkb.embedding.server_side import ServerSideEmbedder
from pkb.embedding.tei_embedder import TEIEmbedder
from pkb.models.config import EmbeddingConfig


class TestCreateEmbedder:
    def test_server_mode_returns_server_side(self):
        config = EmbeddingConfig(mode="server")
        embedder = create_embedder(config)
        assert isinstance(embedder, ServerSideEmbedder)

    def test_tei_mode_returns_tei_embedder(self):
        config = EmbeddingConfig(
            mode="tei",
            model_name="BAAI/bge-m3",
            dimensions=1024,
            tei_url="http://localhost:8090",
        )
        embedder = create_embedder(config)
        assert isinstance(embedder, TEIEmbedder)
        assert embedder.model_name == "BAAI/bge-m3"
        assert embedder.dimensions == 1024

    def test_unknown_mode_raises(self):
        config = EmbeddingConfig(mode="unknown")
        with pytest.raises(ValueError, match="Unknown embedding mode"):
            create_embedder(config)

    def test_default_config_returns_tei(self):
        """Default config is now TEI (bge-m3)."""
        config = EmbeddingConfig()
        embedder = create_embedder(config)
        assert isinstance(embedder, TEIEmbedder)
        assert embedder.model_name == "BAAI/bge-m3"
