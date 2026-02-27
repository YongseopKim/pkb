"""Tests for Embedder ABC."""

import pytest

from pkb.embedding.base import Embedder


class TestEmbedderABC:
    """Embedder는 ABC — 직접 인스턴스화 불가."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Embedder()

    def test_has_embed_documents(self):
        assert hasattr(Embedder, "embed_documents")

    def test_has_embed_query(self):
        assert hasattr(Embedder, "embed_query")

    def test_has_model_name_property(self):
        assert hasattr(Embedder, "model_name")

    def test_has_dimensions_property(self):
        assert hasattr(Embedder, "dimensions")

    def test_subclass_can_implement(self):
        class Dummy(Embedder):
            @property
            def model_name(self) -> str:
                return "dummy"

            @property
            def dimensions(self) -> int:
                return 128

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * 128 for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                return [0.0] * 128

        d = Dummy()
        assert d.model_name == "dummy"
        assert d.dimensions == 128
        assert len(d.embed_documents(["hello"])) == 1
        assert len(d.embed_query("hello")) == 128
