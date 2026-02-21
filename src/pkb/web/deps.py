"""Dependency injection for FastAPI routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pkb.db.chromadb_client import ChunkStore
from pkb.db.postgres import BundleRepository
from pkb.search.engine import SearchEngine

if TYPE_CHECKING:
    from pkb.chat.engine import ChatEngine


class AppState:
    """Holds shared application state (DB connections, services)."""

    def __init__(
        self,
        *,
        repo: BundleRepository,
        chunk_store: ChunkStore,
        search_engine: SearchEngine,
        chat_engine: ChatEngine | None = None,
    ) -> None:
        self.repo = repo
        self.chunk_store = chunk_store
        self.search_engine = search_engine
        self.chat_engine = chat_engine

    def close(self) -> None:
        self.repo.close()
