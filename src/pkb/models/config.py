"""Data models for PKB configuration."""

import os
from pathlib import Path

from pydantic import BaseModel, field_validator


class KBEntry(BaseModel):
    """A knowledge base directory entry."""

    name: str
    path: Path
    watch_dir: Path | None = None

    @field_validator("path", mode="before")
    @classmethod
    def expand_tilde(cls, v: str | Path) -> Path:
        return Path(str(v)).expanduser()

    @field_validator("watch_dir", mode="before")
    @classmethod
    def expand_watch_dir(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        return Path(str(v)).expanduser()

    def get_watch_dir(self) -> Path:
        """Return the watch directory, defaulting to {kb_path}/inbox."""
        return self.watch_dir if self.watch_dir is not None else self.path / "inbox"


class MetaLLMConfig(BaseModel):
    """Configuration for the meta-generation LLM."""

    provider: str = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    max_retries: int = 3
    temperature: float = 0


class EmbeddingConfig(BaseModel):
    """Configuration for text chunking and embedding strategy.

    mode:
        "server" — ChromaDB computes embeddings server-side (default, backward compatible).
        "tei"    — PKB calls a TEI server and sends pre-computed embeddings to ChromaDB.
    """

    chunk_size: int = 1500
    chunk_overlap: int = 200
    mode: str = "tei"
    model_name: str = "BAAI/bge-m3"
    dimensions: int = 1024
    tei_url: str = "http://localhost:8090"
    tei_batch_size: int = 32
    tei_timeout: float = 120.0
    tei_max_concurrent: int = 2


class PostgresConfig(BaseModel):
    """PostgreSQL connection configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "pkb_db"
    username: str = "pkb_user"
    password: str = ""

    def get_dsn(self) -> str:
        """Build a PostgreSQL DSN. PKB_DB_PASSWORD env var takes priority."""
        password = os.environ.get("PKB_DB_PASSWORD", self.password)
        return (
            f"postgresql://{self.username}:{password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class ChromaDBConfig(BaseModel):
    """ChromaDB server connection configuration."""

    host: str = "localhost"
    port: int = 8000
    collection: str = "pkb_chunks"


class DatabaseConfig(BaseModel):
    """Database configuration for PostgreSQL and ChromaDB."""

    postgres: PostgresConfig = PostgresConfig()
    chromadb: ChromaDBConfig = ChromaDBConfig()


class DedupConfig(BaseModel):
    """Configuration for duplicate detection."""

    threshold: float = 0.85


class RelationConfig(BaseModel):
    """Configuration for knowledge graph relation detection."""

    similarity_threshold: float = 0.7
    max_relations_per_bundle: int = 20


class DigestConfig(BaseModel):
    """Configuration for Smart Digest generation."""

    max_bundles: int = 20
    max_tokens: int = 4096


class ConcurrencyConfig(BaseModel):
    """Configuration for concurrent ingest engine."""

    max_concurrent_files: int = 4
    max_concurrent_llm: int = 4
    max_queue_size: int = 10000
    batch_window: float = 5.0
    max_batch_size: int = 50
    chunk_buffer_size: int = 0
    chunk_flush_interval: float = 10.0
    db_pool_min: int = 2
    db_pool_max: int = 8
    retry_interval: float = 300.0  # seconds; 0 to disable periodic retry


class PostIngestConfig(BaseModel):
    """Post-ingest 자동 파이프라인 설정."""

    auto_relate: bool = True
    auto_dedup: bool = True
    gap_update: bool = True


class SchedulerConfig(BaseModel):
    """주기적 자동 작업 설정."""

    weekly_digest: bool = True
    monthly_report: bool = True
    gap_threshold: int = 3


class LLMModelEntry(BaseModel):
    """A model entry with tier assignment.

    When tier is None, the router resolves it from the model catalog.
    Priority: explicit tier > catalog > default (1).
    """

    name: str
    tier: int | None = None


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    api_key_env: str | None = None
    api_key: str | None = None
    models: list[LLMModelEntry] = []


class LLMRoutingConfig(BaseModel):
    """Routing rules: which tier to use for which task."""

    meta_extraction: int = 1
    chat: int = 2
    escalation: bool = True


class LLMConfig(BaseModel):
    """Multi-provider LLM configuration."""

    default_provider: str = "anthropic"
    providers: dict[str, LLMProviderConfig] = {}
    routing: LLMRoutingConfig = LLMRoutingConfig()


class PKBConfig(BaseModel):
    """Top-level PKB configuration (config.yaml)."""

    knowledge_bases: list[KBEntry] = []
    meta_llm: MetaLLMConfig = MetaLLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    database: DatabaseConfig = DatabaseConfig()
    dedup: DedupConfig = DedupConfig()
    relations: RelationConfig = RelationConfig()
    digest: DigestConfig = DigestConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    post_ingest: PostIngestConfig = PostIngestConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    llm: LLMConfig | None = None
