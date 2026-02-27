"""Tests for ServerSideEmbedder sentinel."""

import pytest

from pkb.embedding.server_side import ServerSideEmbedder


class TestServerSideEmbedder:
    """ServerSideEmbedder는 sentinel — embed 호출 시 에러."""

    def test_is_embedder_subclass(self):
        from pkb.embedding.base import Embedder
        e = ServerSideEmbedder()
        assert isinstance(e, Embedder)

    def test_model_name(self):
        e = ServerSideEmbedder()
        assert e.model_name == "server-side"

    def test_dimensions_zero(self):
        e = ServerSideEmbedder()
        assert e.dimensions == 0

    def test_embed_documents_raises(self):
        e = ServerSideEmbedder()
        with pytest.raises(NotImplementedError):
            e.embed_documents(["hello"])

    def test_embed_query_raises(self):
        e = ServerSideEmbedder()
        with pytest.raises(NotImplementedError):
            e.embed_query("hello")
