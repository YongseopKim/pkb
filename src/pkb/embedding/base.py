"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract embedder interface.

    All embedding providers (TEI, server-side sentinel, future providers)
    implement this interface.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding vector dimensionality."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents, returning one vector per document."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text, returning one vector."""
