"""Live LLM integration tests — requires real API keys.

Gate: PKB_LLM_INTEGRATION=1 env var must be set.
Each provider test additionally requires its own API key.
"""

import os

import pytest

from tests.integration.conftest import SKIP_ALL

pytestmark = pytest.mark.skipif(SKIP_ALL, reason="PKB_LLM_INTEGRATION not set")


class TestAnthropicLive:
    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
    def test_complete(self, real_config):
        from pkb.llm.anthropic_provider import AnthropicProvider

        prov_config = real_config.llm.providers.get("anthropic") if real_config.llm else None
        if prov_config is None:
            pytest.skip("anthropic not configured")

        model = prov_config.models[0].name
        provider = AnthropicProvider(model=model, api_key=os.environ["ANTHROPIC_API_KEY"])
        result = provider.complete("Say hello in one word.", max_tokens=10)
        assert len(result) > 0

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
    def test_model_name(self, real_config):
        from pkb.llm.anthropic_provider import AnthropicProvider

        prov_config = real_config.llm.providers.get("anthropic") if real_config.llm else None
        if prov_config is None:
            pytest.skip("anthropic not configured")

        model = prov_config.models[0].name
        provider = AnthropicProvider(model=model)
        assert provider.model_name() == model


class TestOpenAILive:
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
    def test_complete(self, real_config):
        from pkb.llm.openai_provider import OpenAIProvider

        prov_config = real_config.llm.providers.get("openai") if real_config.llm else None
        if prov_config is None:
            pytest.skip("openai not configured")

        model = prov_config.models[0].name
        provider = OpenAIProvider(model=model, api_key=os.environ["OPENAI_API_KEY"])
        result = provider.complete("Say hello in one word.", max_tokens=10)
        assert len(result) > 0


class TestGoogleLive:
    @pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="no GOOGLE_API_KEY")
    def test_complete(self, real_config):
        from pkb.llm.google_provider import GoogleProvider

        prov_config = real_config.llm.providers.get("google") if real_config.llm else None
        if prov_config is None:
            pytest.skip("google not configured")

        model = prov_config.models[0].name
        provider = GoogleProvider(model=model, api_key=os.environ["GOOGLE_API_KEY"])
        result = provider.complete("Say hello in one word.", max_tokens=10)
        assert len(result) > 0


class TestGrokLive:
    @pytest.mark.skipif(not os.environ.get("XAI_API_KEY"), reason="no XAI_API_KEY")
    def test_complete(self, real_config):
        from pkb.llm.grok_provider import GrokProvider

        prov_config = real_config.llm.providers.get("grok") if real_config.llm else None
        if prov_config is None:
            pytest.skip("grok not configured")

        model = prov_config.models[0].name
        provider = GrokProvider(model=model, api_key=os.environ["XAI_API_KEY"])
        result = provider.complete("Say hello in one word.", max_tokens=10)
        assert len(result) > 0


class TestRouterLive:
    def test_router_complete(self, real_router):
        """Test that the router can complete a request via default provider."""
        result = real_router.complete(
            "Say hello in one word.", task="meta_extraction", max_tokens=10,
        )
        assert len(result) > 0

    def test_router_escalation_on_invalid_provider(self, real_config):
        """Test that escalation works when first provider fails."""
        from pkb.models.config import (
            LLMConfig,
            LLMModelEntry,
            LLMProviderConfig,
            LLMRoutingConfig,
        )

        if real_config.llm is None:
            pytest.skip("no llm config")

        # Find a working non-anthropic provider (to avoid dict key collision
        # with the intentionally-broken anthropic entry below).
        working_name = None
        working_config = None
        for name, pconf in real_config.llm.providers.items():
            if name == "anthropic":
                continue
            # Resolve key to verify credentials exist
            api_key = None
            if pconf.api_key_env:
                api_key = os.environ.get(pconf.api_key_env)
            if not api_key and pconf.api_key:
                api_key = pconf.api_key
            if api_key:
                working_name = name
                working_config = pconf
                break

        if working_name is None:
            pytest.skip("need at least 1 working non-anthropic provider")

        from pkb.llm.router import LLMRouter

        config = LLMConfig(
            default_provider="anthropic",
            providers={
                "anthropic": LLMProviderConfig(
                    api_key="sk-invalid-key-for-escalation-test",
                    models=[LLMModelEntry(name="claude-haiku-4-5-20251001", tier=1)],
                ),
                working_name: LLMProviderConfig(
                    api_key_env=working_config.api_key_env,
                    api_key=working_config.api_key,
                    models=[
                        LLMModelEntry(name=working_config.models[0].name, tier=1),
                    ],
                ),
            },
            routing=LLMRoutingConfig(meta_extraction=1, chat=1, escalation=True),
        )

        router = LLMRouter(config)
        # Should escalate past the invalid anthropic key to the working provider.
        # Use max_tokens=200 because reasoning models (e.g. gpt-5-mini) consume
        # tokens internally for chain-of-thought and need headroom to produce output.
        result = router.complete("Say hello in one word.", max_tokens=200)
        assert len(result) > 0


class TestAPIKeyPriority:
    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
    def test_env_takes_priority_over_config(self, real_config):
        """When both env and config have keys, env wins."""
        from pkb.doctor import DoctorRunner

        doctor = DoctorRunner(pkb_home=os.path.expanduser("~/.pkb"))
        key, source = doctor._resolve_api_key(
            "anthropic", "ANTHROPIC_API_KEY", "sk-from-config",
        )
        assert source == "env ANTHROPIC_API_KEY"
        assert key == os.environ["ANTHROPIC_API_KEY"]

    def test_config_key_when_no_env(self, monkeypatch):
        """When env var is not set, config.yaml key is used."""
        from pkb.doctor import DoctorRunner

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        doctor = DoctorRunner(pkb_home=os.path.expanduser("~/.pkb"))
        key, source = doctor._resolve_api_key(
            "anthropic", "ANTHROPIC_API_KEY", "sk-from-config",
        )
        assert source == "config.yaml"
        assert key == "sk-from-config"

    def test_sdk_default_when_no_key(self, monkeypatch):
        """When neither env nor config has a key, falls back to SDK default."""
        from pkb.doctor import DoctorRunner

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        doctor = DoctorRunner(pkb_home=os.path.expanduser("~/.pkb"))
        key, source = doctor._resolve_api_key("anthropic", "ANTHROPIC_API_KEY", None)
        assert source == "SDK default"
        assert key is None
