"""Factory for creating embedder instances from config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pkb.embedding.base import Embedder

if TYPE_CHECKING:
    from pkb.models.config import EmbeddingConfig


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Create an Embedder based on config.mode.

    "server" → ServerSideEmbedder (ChromaDB handles embeddings)
    "tei"    → TEIEmbedder (PKB calls TEI server)
    """
    if config.mode == "server":
        from pkb.embedding.server_side import ServerSideEmbedder
        return ServerSideEmbedder()

    if config.mode == "tei":
        from pkb.embedding.tei_client import TEIClient
        from pkb.embedding.tei_embedder import TEIEmbedder

        client = TEIClient(
            base_url=config.tei_url,
            timeout=config.tei_timeout,
            max_concurrent=config.tei_max_concurrent,
        )
        return TEIEmbedder(
            client=client,
            model_name=config.model_name,
            dimensions=config.dimensions,
            batch_size=config.tei_batch_size,
        )

    raise ValueError(f"Unknown embedding mode: {config.mode!r}")
