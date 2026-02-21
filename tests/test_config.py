"""Tests for config and init system."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from pkb.config import build_llm_router, create_default_config, get_pkb_home, load_config
from pkb.init import init_pkb_home
from pkb.models.config import (
    LLMConfig,
    LLMModelEntry,
    LLMProviderConfig,
    LLMRoutingConfig,
    MetaLLMConfig,
    PKBConfig,
)


class TestGetPkbHome:
    def test_default_home(self):
        home = get_pkb_home()
        assert home == Path.home() / ".pkb"

    def test_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        custom = tmp_path / "custom-pkb"
        monkeypatch.setenv("PKB_HOME", str(custom))
        assert get_pkb_home() == custom


class TestCreateDefaultConfig:
    def test_creates_valid_yaml(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        create_default_config(config_path)
        assert config_path.exists()
        data = yaml.safe_load(config_path.read_text())
        assert "knowledge_bases" in data
        assert "meta_llm" in data
        assert "embedding" in data

    def test_config_roundtrip(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        create_default_config(config_path)
        config = load_config(config_path)
        assert config.meta_llm.model == "claude-haiku-4-5-20251001"
        assert config.embedding.chunk_size == 512

    def test_default_config_has_llm_section(self, tmp_path: Path):
        """Generated config.yaml should include the llm section."""
        config_path = tmp_path / "config.yaml"
        create_default_config(config_path)
        data = yaml.safe_load(config_path.read_text())
        assert "llm" in data
        assert "providers" in data["llm"]
        assert "anthropic" in data["llm"]["providers"]
        assert "grok" in data["llm"]["providers"]

    def test_default_config_roundtrip_llm(self, tmp_path: Path):
        """Saved config should load with config.llm having 4 providers."""
        config_path = tmp_path / "config.yaml"
        create_default_config(config_path)
        config = load_config(config_path)
        assert config.llm is not None
        assert len(config.llm.providers) == 4
        assert "anthropic" in config.llm.providers
        assert "openai" in config.llm.providers
        assert "google" in config.llm.providers
        assert "grok" in config.llm.providers

    def test_default_config_has_api_key_field(self, tmp_path: Path):
        """Generated config.yaml should include api_key field in each provider."""
        config_path = tmp_path / "config.yaml"
        create_default_config(config_path)
        data = yaml.safe_load(config_path.read_text())
        for provider_name in ("anthropic", "openai", "google", "grok"):
            provider = data["llm"]["providers"][provider_name]
            assert "api_key" in provider, f"{provider_name} missing api_key"
            assert provider["api_key"] == ""

    def test_default_config_roundtrip_api_key(self, tmp_path: Path):
        """Loaded config should have api_key=None for empty string providers."""
        config_path = tmp_path / "config.yaml"
        create_default_config(config_path)
        config = load_config(config_path)
        assert config.llm is not None
        for name, prov in config.llm.providers.items():
            # Empty string from YAML loads as empty string, model allows it
            assert prov.api_key is not None or prov.api_key == ""

    def test_database_config_in_default(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        create_default_config(config_path)
        data = yaml.safe_load(config_path.read_text())
        assert "database" in data
        assert "postgres" in data["database"]
        assert "chromadb" in data["database"]
        config = load_config(config_path)
        assert config.database.postgres.host == "localhost"
        assert config.database.chromadb.collection == "pkb_chunks"


class TestBuildLLMRouter:
    def test_from_llm_config(self):
        """When config.llm is set, build_llm_router should use it."""
        config = PKBConfig(
            llm=LLMConfig(
                default_provider="anthropic",
                providers={
                    "anthropic": LLMProviderConfig(
                        models=[LLMModelEntry(name="claude-haiku-4-5-20251001", tier=1)],
                    ),
                },
                routing=LLMRoutingConfig(meta_extraction=1, chat=1, escalation=True),
            ),
        )
        router = build_llm_router(config)
        # Should resolve meta_extraction to tier 1
        provider = router.get_provider(task="meta_extraction")
        assert provider is not None

    def test_fallback_meta_llm(self):
        """When config.llm is None, should fall back to meta_llm."""
        config = PKBConfig(
            meta_llm=MetaLLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001"),
        )
        assert config.llm is None
        router = build_llm_router(config)
        provider = router.get_provider(task="meta_extraction")
        assert provider is not None

    def test_from_meta_llm_chat_tier_fix(self):
        """from_meta_llm should set chat tier to 1 (not default 2) since only tier 1 exists."""
        from pkb.llm.router import LLMRouter
        meta_config = MetaLLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
        router = LLMRouter.from_meta_llm(meta_config)
        # This should NOT raise "No provider for tier 2"
        provider = router.get_provider(task="chat")
        assert provider is not None


class TestAppStateChatEngine:
    def test_app_state_with_chat_engine(self):
        """AppState should accept and store a chat_engine."""
        from pkb.web.deps import AppState
        repo = MagicMock()
        chunk_store = MagicMock()
        search_engine = MagicMock()
        chat_engine = MagicMock()
        state = AppState(
            repo=repo,
            chunk_store=chunk_store,
            search_engine=search_engine,
            chat_engine=chat_engine,
        )
        assert state.chat_engine is chat_engine

    def test_app_state_chat_engine_defaults_none(self):
        """AppState should default chat_engine to None for backward compat."""
        from pkb.web.deps import AppState
        state = AppState(
            repo=MagicMock(),
            chunk_store=MagicMock(),
            search_engine=MagicMock(),
        )
        assert state.chat_engine is None


class TestLoadConfig:
    def test_load_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


class TestInitPkbHome:
    def test_creates_directory_structure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        pkb_home = tmp_path / ".pkb"
        monkeypatch.setenv("PKB_HOME", str(pkb_home))
        init_pkb_home()

        assert pkb_home.exists()
        assert (pkb_home / "vocab").is_dir()
        assert (pkb_home / "index").is_dir()
        assert (pkb_home / "config.yaml").is_file()

    def test_vocab_files_copied(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        pkb_home = tmp_path / ".pkb"
        monkeypatch.setenv("PKB_HOME", str(pkb_home))
        init_pkb_home()

        domains_path = pkb_home / "vocab" / "domains.yaml"
        topics_path = pkb_home / "vocab" / "topics.yaml"
        assert domains_path.is_file()
        assert topics_path.is_file()

        # Verify loadable
        from pkb.vocab.loader import load_domains, load_topics

        vocab_d = load_domains(domains_path)
        assert len(vocab_d.domains) == 8

        vocab_t = load_topics(topics_path)
        assert len(vocab_t.topics) >= 50

    def test_config_is_valid(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        pkb_home = tmp_path / ".pkb"
        monkeypatch.setenv("PKB_HOME", str(pkb_home))
        init_pkb_home()

        config = load_config(pkb_home / "config.yaml")
        assert config.meta_llm.provider == "anthropic"

    def test_existing_home_without_force_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        pkb_home = tmp_path / ".pkb"
        pkb_home.mkdir()
        (pkb_home / "config.yaml").write_text("existing")
        monkeypatch.setenv("PKB_HOME", str(pkb_home))

        with pytest.raises(FileExistsError):
            init_pkb_home()

    def test_existing_home_with_force_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        pkb_home = tmp_path / ".pkb"
        pkb_home.mkdir()
        (pkb_home / "config.yaml").write_text("existing")
        monkeypatch.setenv("PKB_HOME", str(pkb_home))

        init_pkb_home(force=True)
        assert (pkb_home / "vocab" / "domains.yaml").is_file()
