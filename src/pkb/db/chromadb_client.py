"""ChromaDB client for PKB vector storage."""

from dataclasses import dataclass

import chromadb

from pkb.models.config import ChromaDBConfig


@dataclass
class SearchResult:
    """A single search result from ChromaDB."""

    chunk_id: str
    document: str
    metadata: dict
    distance: float


class ChunkStore:
    """Vector store for text chunks using ChromaDB."""

    def __init__(self, config: ChromaDBConfig) -> None:
        self._client = chromadb.HttpClient(host=config.host, port=config.port)
        self._collection = self._client.get_or_create_collection(
            name=config.collection,
        )

    def upsert_chunks(self, chunks: list[dict]) -> None:
        """Upsert chunks into ChromaDB.

        Each chunk dict must have: id, document, metadata.
        ChromaDB server computes embeddings from documents.
        """
        if not chunks:
            return
        self._collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["document"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
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
        """Search for similar chunks. ChromaDB computes query embedding server-side."""
        kwargs: dict = {"query_texts": [query], "n_results": n_results}
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

    def heartbeat(self) -> int:
        """Check ChromaDB server connectivity."""
        return self._client.heartbeat()
