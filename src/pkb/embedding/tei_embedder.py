"""TEI-based embedder implementation."""

from __future__ import annotations

from pkb.embedding.base import Embedder
from pkb.embedding.tei_client import TEIClient


class TEIEmbedder(Embedder):
    """Embedder that delegates to a TEI server via TEIClient."""

    def __init__(
        self,
        client: TEIClient,
        model_name: str,
        dimensions: int,
        batch_size: int = 32,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._dimensions = dimensions
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in batches."""
        if not texts:
            return []
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            all_vectors.extend(self._client.embed(batch))
        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        vectors = self._client.embed([text])
        return vectors[0]
