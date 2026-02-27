"""Server-side embedder sentinel.

When mode=server, ChromaDB handles embeddings internally.
This sentinel satisfies the Embedder interface but raises on actual embed calls.
"""

from pkb.embedding.base import Embedder


class ServerSideEmbedder(Embedder):
    """Sentinel: embedding is handled by the vector DB server."""

    @property
    def model_name(self) -> str:
        return "server-side"

    @property
    def dimensions(self) -> int:
        return 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "ServerSideEmbedder does not compute embeddings — "
            "ChromaDB handles them server-side."
        )

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError(
            "ServerSideEmbedder does not compute embeddings — "
            "ChromaDB handles them server-side."
        )
