"""ChromaDB client for PKB vector storage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chromadb

from pkb.models.config import ChromaDBConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pkb.embedding.base import Embedder


@dataclass
class SearchResult:
    """A single search result from ChromaDB."""

    chunk_id: str
    document: str
    metadata: dict
    distance: float


class ChunkStore:
    """Vector store for text chunks using ChromaDB.

    When *embedder* is ``None`` or a ``ServerSideEmbedder``, ChromaDB computes
    embeddings server-side (backward-compatible default).

    When a real ``Embedder`` (e.g. ``TEIEmbedder``) is provided, PKB computes
    embeddings client-side and sends them pre-computed to ChromaDB.
    """

    def __init__(
        self,
        config: ChromaDBConfig,
        embedder: Embedder | None = None,
    ) -> None:
        self._config = config
        self._client = chromadb.HttpClient(host=config.host, port=config.port)
        self._embedder = embedder
        self._use_client_side = self._should_use_client_side(embedder)

        create_kwargs: dict = {"name": config.collection}
        if self._use_client_side and embedder is not None:
            create_kwargs["metadata"] = {
                "embedding_model": embedder.model_name,
                "embedding_dimensions": embedder.dimensions,
                "hnsw:space": "cosine",
            }
            create_kwargs["embedding_function"] = None

        self._collection = self._client.get_or_create_collection(**create_kwargs)
        self._check_model_consistency()

    def _check_model_consistency(self) -> None:
        """Warn if collection's embedding model doesn't match the configured embedder."""
        if not self._use_client_side or self._embedder is None:
            return

        metadata = self._collection.metadata
        if not metadata or "embedding_model" not in metadata:
            logger.warning(
                "Collection '%s' has no embedding model metadata (legacy collection). "
                "Run 'pkb reembed --all --fresh' to re-embed with the current model.",
                self._config.collection,
            )
            return

        stored_model = metadata.get("embedding_model", "")
        if stored_model != self._embedder.model_name:
            logger.warning(
                "Embedding model mismatch: collection has '%s' but config specifies '%s'. "
                "Run 'pkb reembed --all --fresh' to re-embed with the new model.",
                stored_model,
                self._embedder.model_name,
            )

    @staticmethod
    def _should_use_client_side(embedder: Embedder | None) -> bool:
        """Determine whether to use client-side embedding."""
        if embedder is None:
            return False
        from pkb.embedding.server_side import ServerSideEmbedder
        return not isinstance(embedder, ServerSideEmbedder)

    def upsert_chunks(self, chunks: list[dict]) -> None:
        """Upsert chunks into ChromaDB.

        Each chunk dict must have: id, document, metadata.
        Client-side mode sends pre-computed embeddings; server-side sends documents.
        """
        if not chunks:
            return

        ids = [c["id"] for c in chunks]
        documents = [c["document"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        if self._use_client_side and self._embedder is not None:
            embeddings = self._embedder.embed_documents(documents)
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

    def delete_by_bundle(self, bundle_id: str) -> None:
        """Delete all chunks belonging to a bundle."""
        self._collection.delete(where={"bundle_id": bundle_id})

    def delete_by_kb(self, kb: str) -> None:
        """Delete all chunks belonging to a KB."""
        self._collection.delete(where={"kb": kb})

    def search(
        self, query: str, n_results: int = 10, where: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks."""
        kwargs: dict = {"n_results": n_results}

        if self._use_client_side and self._embedder is not None:
            kwargs["query_embeddings"] = [self._embedder.embed_query(query)]
        else:
            kwargs["query_texts"] = [query]

        if where:
            kwargs["where"] = where
        raw = self._collection.query(**kwargs)
        results = []
        if raw["ids"] and raw["ids"][0]:
            for i, chunk_id in enumerate(raw["ids"][0]):
                results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        document=raw["documents"][0][i],
                        metadata=raw["metadatas"][0][i],
                        distance=raw["distances"][0][i],
                    )
                )
        return results

    def get_collection_model_info(self) -> dict:
        """Return embedding model metadata from the collection."""
        metadata = self._collection.metadata
        if metadata is None:
            return {}
        return dict(metadata)

    def drop_and_recreate_collection(self) -> None:
        """Delete the collection and recreate it (for model migration)."""
        self._client.delete_collection(name=self._config.collection)

        create_kwargs: dict = {"name": self._config.collection}
        if self._use_client_side and self._embedder is not None:
            create_kwargs["metadata"] = {
                "embedding_model": self._embedder.model_name,
                "embedding_dimensions": self._embedder.dimensions,
                "hnsw:space": "cosine",
            }
            create_kwargs["embedding_function"] = None

        self._collection = self._client.get_or_create_collection(**create_kwargs)

    def heartbeat(self) -> int:
        """Check ChromaDB server connectivity."""
        return self._client.heartbeat()
