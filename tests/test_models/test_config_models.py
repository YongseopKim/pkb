"""Tests for config data models."""

from pathlib import Path

import pytest

from pkb.models.config import (
    ChromaDBConfig,
    ConcurrencyConfig,
    DatabaseConfig,
    EmbeddingConfig,
    KBEntry,
    LLMConfig,
    LLMModelEntry,
    LLMProviderConfig,
    MetaLLMConfig,
    PKBConfig,
    PostgresConfig,
)


class TestKBEntry:
    def test_valid_entry(self):
        entry = KBEntry(name="personal", path="~/kb-personal")
        assert entry.name == "personal"
        assert entry.path == Path.home() / "kb-personal"

    def test_tilde_expansion(self):
        entry = KBEntry(name="work", path="~/kb-work")
        assert "~" not in str(entry.path)
        assert entry.path.is_absolute()

    def test_absolute_path(self):
        entry = KBEntry(name="test", path="/tmp/kb-test")
        assert entry.path == Path("/tmp/kb-test")

    def test_watch_dir_default_none(self):
        entry = KBEntry(name="personal", path="~/kb-personal")
        assert entry.watch_dir is None

    def test_watch_dir_custom(self):
        entry = KBEntry(name="personal", path="~/kb-personal", watch_dir="~/inbox")
        assert entry.watch_dir is not None
        assert "~" not in str(entry.watch_dir)

    def test_get_watch_dir_default(self):
        entry = KBEntry(name="personal", path="~/kb-personal")
        watch = entry.get_watch_dir()
        assert watch == entry.path / "inbox"

    def test_get_watch_dir_custom(self):
        entry = KBEntry(name="personal", path="~/kb-personal", watch_dir="/tmp/custom")
        watch = entry.get_watch_dir()
        assert watch == Path("/tmp/custom")


class TestMetaLLMConfig:
    def test_defaults(self):
        config = MetaLLMConfig()
        assert config.provider == "anthropic"
        assert config.model == "claude-haiku-4-5-20251001"
        assert config.max_retries == 3
        assert config.temperature == 0


class TestEmbeddingConfig:
    def test_defaults(self):
        config = EmbeddingConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50

    def test_no_model_field(self):
        """EmbeddingConfig no longer has a model field (server-side embedding)."""
        config = EmbeddingConfig()
        assert not hasattr(config, "model")


class TestPostgresConfig:
    def test_defaults(self):
        config = PostgresConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "pkb_db"
        assert config.username == "pkb_user"
        assert config.password == ""

    def test_dsn(self):
        config = PostgresConfig(host="192.168.1.100", password="secret")
        dsn = config.get_dsn()
        assert "192.168.1.100" in dsn
        assert "secret" in dsn
        assert "pkb_db" in dsn

    def test_dsn_env_password_override(self, monkeypatch: pytest.MonkeyPatch):
        """PKB_DB_PASSWORD env var should override config password."""
        monkeypatch.setenv("PKB_DB_PASSWORD", "env_secret")
        config = PostgresConfig(password="config_secret")
        dsn = config.get_dsn()
        assert "env_secret" in dsn
        assert "config_secret" not in dsn

    def test_custom_values(self):
        config = PostgresConfig(
            host="10.0.0.5",
            port=5433,
            database="my_db",
            username="my_user",
            password="my_pass",
        )
        assert config.host == "10.0.0.5"
        assert config.port == 5433
        assert config.database == "my_db"
        assert config.username == "my_user"


class TestChromaDBConfig:
    def test_defaults(self):
        config = ChromaDBConfig()
        assert config.host == "localhost"
        assert config.port == 8000
        assert config.collection == "pkb_chunks"

    def test_custom_values(self):
        config = ChromaDBConfig(host="192.168.1.100", port=9000, collection="my_col")
        assert config.host == "192.168.1.100"
        assert config.port == 9000
        assert config.collection == "my_col"


class TestDatabaseConfig:
    def test_defaults(self):
        config = DatabaseConfig()
        assert config.postgres.host == "localhost"
        assert config.chromadb.port == 8000

    def test_nested_construction(self):
        config = DatabaseConfig(
            postgres=PostgresConfig(host="10.0.0.1"),
            chromadb=ChromaDBConfig(host="10.0.0.2"),
        )
        assert config.postgres.host == "10.0.0.1"
        assert config.chromadb.host == "10.0.0.2"


class TestPKBConfig:
    def test_default_config(self):
        config = PKBConfig()
        assert config.knowledge_bases == []
        assert config.meta_llm.provider == "anthropic"
        assert config.embedding.chunk_size == 512

    def test_database_config_present(self):
        config = PKBConfig()
        assert config.database.postgres.host == "localhost"
        assert config.database.chromadb.collection == "pkb_chunks"

    def test_with_kb_entries(self):
        config = PKBConfig(
            knowledge_bases=[
                KBEntry(name="personal", path="~/kb-personal"),
                KBEntry(name="work", path="~/kb-work"),
            ]
        )
        assert len(config.knowledge_bases) == 2
        assert config.knowledge_bases[0].name == "personal"

    def test_full_config_construction(self):
        config = PKBConfig(
            database=DatabaseConfig(
                postgres=PostgresConfig(host="10.0.0.1"),
                chromadb=ChromaDBConfig(host="10.0.0.1"),
            ),
            embedding=EmbeddingConfig(chunk_size=1024),
        )
        assert config.database.postgres.host == "10.0.0.1"
        assert config.embedding.chunk_size == 1024

    def test_llm_defaults_none(self):
        """PKBConfig.llm should default to None (backward compat)."""
        config = PKBConfig()
        assert config.llm is None

    def test_with_llm_section(self):
        """PKBConfig should accept an llm dict."""
        config = PKBConfig(
            llm=LLMConfig(
                default_provider="anthropic",
                providers={
                    "anthropic": LLMProviderConfig(
                        api_key_env="ANTHROPIC_API_KEY",
                        models=[LLMModelEntry(name="claude-haiku-4-5-20251001", tier=1)],
                    ),
                },
            ),
        )
        assert config.llm is not None
        assert config.llm.default_provider == "anthropic"
        assert len(config.llm.providers) == 1

    def test_backward_compat_no_llm_key(self):
        """Config without llm key should still load (existing config.yaml files)."""
        config = PKBConfig(**{
            "knowledge_bases": [],
            "meta_llm": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        })
        assert config.llm is None
        assert config.meta_llm.provider == "anthropic"


class TestConcurrencyConfig:
    """Tests for ConcurrencyConfig model."""

    def test_defaults(self):
        config = ConcurrencyConfig()
        assert config.max_concurrent_files == 4
        assert config.max_concurrent_llm == 4
        assert config.max_queue_size == 10000
        assert config.batch_window == 5.0
        assert config.max_batch_size == 50
        assert config.chunk_buffer_size == 0
        assert config.chunk_flush_interval == 10.0
        assert config.db_pool_min == 2
        assert config.db_pool_max == 8

    def test_custom_values(self):
        config = ConcurrencyConfig(
            max_concurrent_files=8,
            max_concurrent_llm=2,
            max_queue_size=5000,
            batch_window=10.0,
        )
        assert config.max_concurrent_files == 8
        assert config.max_concurrent_llm == 2
        assert config.max_queue_size == 5000
        assert config.batch_window == 10.0

    def test_pkbconfig_concurrency_default(self):
        """PKBConfig.concurrency should default to ConcurrencyConfig()."""
        config = PKBConfig()
        assert config.concurrency is not None
        assert config.concurrency.max_concurrent_files == 4

    def test_pkbconfig_backward_compat_no_concurrency_key(self):
        """Config without concurrency key should still load."""
        config = PKBConfig(**{
            "knowledge_bases": [],
            "meta_llm": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        })
        assert config.concurrency.max_concurrent_files == 4

    def test_pkbconfig_with_concurrency(self):
        """Config with concurrency key should override defaults."""
        config = PKBConfig(**{
            "concurrency": {"max_concurrent_files": 16, "batch_window": 3.0},
        })
        assert config.concurrency.max_concurrent_files == 16
        assert config.concurrency.batch_window == 3.0
        # Other fields keep defaults
        assert config.concurrency.max_queue_size == 10000


class TestLLMProviderConfigApiKey:
    """Tests for LLMProviderConfig.api_key field."""

    def test_api_key_defaults_none(self):
        """api_key should default to None."""
        config = LLMProviderConfig()
        assert config.api_key is None

    def test_api_key_set_directly(self):
        """api_key should accept a direct string value."""
        config = LLMProviderConfig(api_key="sk-test-123")
        assert config.api_key == "sk-test-123"

    def test_api_key_with_api_key_env(self):
        """api_key and api_key_env can coexist."""
        config = LLMProviderConfig(
            api_key_env="ANTHROPIC_API_KEY",
            api_key="sk-direct-key",
        )
        assert config.api_key_env == "ANTHROPIC_API_KEY"
        assert config.api_key == "sk-direct-key"

    def test_api_key_empty_string_treated_as_set(self):
        """Empty string api_key should be preserved (user explicitly set it)."""
        config = LLMProviderConfig(api_key="")
        assert config.api_key == ""

    def test_api_key_from_yaml_dict(self):
        """api_key should parse from dict (YAML deserialization)."""
        data = {
            "api_key_env": "OPENAI_API_KEY",
            "api_key": "sk-yaml-key",
            "models": [{"name": "gpt-4o-mini", "tier": 1}],
        }
        config = LLMProviderConfig(**data)
        assert config.api_key == "sk-yaml-key"
        assert len(config.models) == 1


class TestLLMModelEntryTier:
    """Tests for LLMModelEntry.tier optional behavior."""

    def test_tier_defaults_none(self):
        """LLMModelEntry without tier should default to None."""
        entry = LLMModelEntry(name="test-model")
        assert entry.tier is None

    def test_tier_explicit_preserved(self):
        """Explicitly set tier should be preserved."""
        entry = LLMModelEntry(name="test-model", tier=2)
        assert entry.tier == 2

    def test_tier_from_dict_without_tier(self):
        """Dict without tier key should produce tier=None."""
        entry = LLMModelEntry(**{"name": "test-model"})
        assert entry.tier is None

    def test_tier_from_dict_with_tier(self):
        """Dict with tier key should preserve the value."""
        entry = LLMModelEntry(**{"name": "test-model", "tier": 1})
        assert entry.tier == 1
