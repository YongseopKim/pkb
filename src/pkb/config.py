"""PKB configuration management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from pkb.constants import DEFAULT_PKB_HOME
from pkb.models.config import PKBConfig

if TYPE_CHECKING:
    from pkb.db.chromadb_client import ChunkStore
    from pkb.llm.router import LLMRouter


def get_pkb_home() -> Path:
    """Get PKB home directory. Respects PKB_HOME env variable."""
    env_home = os.environ.get("PKB_HOME")
    if env_home:
        return Path(env_home)
    return DEFAULT_PKB_HOME


def load_config(path: Path) -> PKBConfig:
    """Load PKB config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PKBConfig(**raw)


def build_llm_router(config: PKBConfig) -> LLMRouter:
    """Build an LLMRouter from PKBConfig.

    Uses config.llm if present, otherwise falls back to legacy config.meta_llm.
    """
    from pkb.llm.router import LLMRouter as _LLMRouter

    if config.llm is not None:
        return _LLMRouter(config.llm)
    return _LLMRouter.from_meta_llm(config.meta_llm)


def build_chunk_store(config: PKBConfig) -> ChunkStore:
    """Build a ChunkStore with the appropriate embedder based on config."""
    from pkb.db.chromadb_client import ChunkStore as _ChunkStore
    from pkb.embedding.factory import create_embedder

    embedder = create_embedder(config.embedding)

    from pkb.embedding.server_side import ServerSideEmbedder
    if isinstance(embedder, ServerSideEmbedder):
        return _ChunkStore(config.database.chromadb)

    return _ChunkStore(config.database.chromadb, embedder=embedder)


def create_default_config(path: Path) -> None:
    """Create a default config.yaml file."""
    config = PKBConfig()
    data = {
        "knowledge_bases": [],
        "meta_llm": {
            "provider": config.meta_llm.provider,
            "model": config.meta_llm.model,
            "max_retries": config.meta_llm.max_retries,
            "temperature": config.meta_llm.temperature,
        },
        "embedding": {
            "chunk_size": config.embedding.chunk_size,
            "chunk_overlap": config.embedding.chunk_overlap,
            "mode": config.embedding.mode,
            "model_name": config.embedding.model_name,
            "dimensions": config.embedding.dimensions,
            "tei_url": config.embedding.tei_url,
            "tei_batch_size": config.embedding.tei_batch_size,
            "tei_timeout": config.embedding.tei_timeout,
        },
        "database": {
            "postgres": {
                "host": config.database.postgres.host,
                "port": config.database.postgres.port,
                "database": config.database.postgres.database,
                "username": config.database.postgres.username,
                "password": "",
            },
            "chromadb": {
                "host": config.database.chromadb.host,
                "port": config.database.chromadb.port,
                "collection": config.database.chromadb.collection,
            },
        },
        "llm": {
            "default_provider": "anthropic",
            "providers": {
                "anthropic": {
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "api_key": "",
                    "models": [{"name": "claude-haiku-4-5-20251001", "tier": 1}],
                },
                "openai": {
                    "api_key_env": "OPENAI_API_KEY",
                    "api_key": "",
                    "models": [{"name": "gpt-4o-mini", "tier": 1}],
                },
                "google": {
                    "api_key_env": "GOOGLE_API_KEY",
                    "api_key": "",
                    "models": [{"name": "gemini-2.0-flash", "tier": 1}],
                },
                "grok": {
                    "api_key_env": "XAI_API_KEY",
                    "api_key": "",
                    "models": [{"name": "grok-3-mini-fast", "tier": 1}],
                },
            },
            "routing": {
                "meta_extraction": 1,
                "chat": 1,
                "escalation": True,
            },
        },
    }
    path.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False), encoding="utf-8"
    )
