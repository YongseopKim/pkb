"""Tests for LLM Router."""

from unittest.mock import MagicMock, patch

import pytest

from pkb.llm.router import LLMRouter
from pkb.models.config import (
    LLMConfig,
    LLMModelEntry,
    LLMProviderConfig,
    LLMRoutingConfig,
)


class TestLLMConfig:
    def test_default_config(self):
        config = LLMConfig()
        assert config.default_provider == "anthropic"

    def test_config_with_providers(self):
        config = LLMConfig(
            providers={
                "anthropic": LLMProviderConfig(
                    models=[LLMModelEntry(name="claude-haiku-4-5-20251001", tier=1)]
                ),
            }
        )
        assert len(config.providers["anthropic"].models) == 1

    def test_routing_config(self):
        config = LLMRoutingConfig()
        assert config.meta_extraction == 1
        assert config.chat == 2


class TestLLMRouter:
    def test_get_provider_for_task(self):
        """Router should select correct tier based on task."""
        mock_provider = MagicMock()
        mock_provider.complete.return_value = "result"
        mock_provider.model_name.return_value = "test-model"

        router = LLMRouter.__new__(LLMRouter)
        router._providers = {"anthropic": [mock_provider]}
        router._tier_map = {1: [("anthropic", mock_provider)], 2: []}
        router._routing = LLMRoutingConfig()
        router._escalation = True

        provider = router.get_provider(task="meta_extraction")
        assert provider is mock_provider

    def test_complete_delegates_to_provider(self):
        mock_provider = MagicMock()
        mock_provider.complete.return_value = "LLM response"

        router = LLMRouter.__new__(LLMRouter)
        router._providers = {"anthropic": [mock_provider]}
        router._tier_map = {1: [("anthropic", mock_provider)]}
        router._routing = LLMRoutingConfig()
        router._escalation = True

        result = router.complete("prompt", task="meta_extraction", max_tokens=1024)
        assert result == "LLM response"
        mock_provider.complete.assert_called_once()

    def test_escalation_on_failure(self):
        """When first provider fails, escalate to next in same tier."""
        failing_provider = MagicMock()
        failing_provider.complete.side_effect = Exception("API error")
        failing_provider.model_name.return_value = "model-a"

        backup_provider = MagicMock()
        backup_provider.complete.return_value = "backup result"
        backup_provider.model_name.return_value = "model-b"

        router = LLMRouter.__new__(LLMRouter)
        router._providers = {
            "anthropic": [failing_provider],
            "openai": [backup_provider],
        }
        router._tier_map = {
            1: [("anthropic", failing_provider), ("openai", backup_provider)],
        }
        router._routing = LLMRoutingConfig()
        router._escalation = True

        result = router.complete("prompt", task="meta_extraction", max_tokens=100)
        assert result == "backup result"

    def test_escalation_disabled_raises(self):
        """When escalation is disabled, failure should raise."""
        failing_provider = MagicMock()
        failing_provider.complete.side_effect = Exception("API error")
        failing_provider.model_name.return_value = "model-a"

        router = LLMRouter.__new__(LLMRouter)
        router._providers = {"anthropic": [failing_provider]}
        router._tier_map = {1: [("anthropic", failing_provider)]}
        router._routing = LLMRoutingConfig()
        router._escalation = False

        with pytest.raises(Exception, match="API error"):
            router.complete("prompt", task="meta_extraction", max_tokens=100)

    def test_no_provider_for_tier_raises(self):
        """Should raise when no provider is configured for the requested tier."""
        router = LLMRouter.__new__(LLMRouter)
        router._providers = {}
        router._tier_map = {}
        router._routing = LLMRoutingConfig()
        router._escalation = True

        with pytest.raises(ValueError, match="No provider"):
            router.get_provider(task="meta_extraction")


class TestCrossTierEscalation:
    """Cross-tier escalation: tier 1 fails → try tier 2 → tier 3."""

    def test_cross_tier_on_failure(self):
        """tier 1 fails → escalate to tier 2."""
        failing_t1 = MagicMock()
        failing_t1.complete.side_effect = Exception("tier 1 down")
        success_t2 = MagicMock()
        success_t2.complete.return_value = "tier 2 result"

        router = LLMRouter.__new__(LLMRouter)
        router._providers = {}
        router._tier_map = {
            1: [("anthropic", failing_t1)],
            2: [("openai", success_t2)],
        }
        router._routing = LLMRoutingConfig(meta_extraction=1)
        router._escalation = True

        result = router.complete("prompt", task="meta_extraction")
        assert result == "tier 2 result"

    def test_cross_tier_all_providers_per_tier(self):
        """Both tier 1 providers fail → escalate to tier 2."""
        fail_a = MagicMock()
        fail_a.complete.side_effect = Exception("A down")
        fail_b = MagicMock()
        fail_b.complete.side_effect = Exception("B down")
        success_c = MagicMock()
        success_c.complete.return_value = "C ok"

        router = LLMRouter.__new__(LLMRouter)
        router._providers = {}
        router._tier_map = {
            1: [("anthropic", fail_a), ("openai", fail_b)],
            2: [("google", success_c)],
        }
        router._routing = LLMRoutingConfig(meta_extraction=1)
        router._escalation = True

        result = router.complete("prompt", task="meta_extraction")
        assert result == "C ok"

    def test_cross_tier_disabled_stays_same_tier(self):
        """escalation=False → no cross-tier, raise on first failure."""
        failing_t1 = MagicMock()
        failing_t1.complete.side_effect = Exception("tier 1 down")
        success_t2 = MagicMock()
        success_t2.complete.return_value = "tier 2"

        router = LLMRouter.__new__(LLMRouter)
        router._providers = {}
        router._tier_map = {
            1: [("anthropic", failing_t1)],
            2: [("openai", success_t2)],
        }
        router._routing = LLMRoutingConfig(meta_extraction=1)
        router._escalation = False

        with pytest.raises(Exception, match="tier 1 down"):
            router.complete("prompt", task="meta_extraction")

    def test_cross_tier_skips_lower_tiers(self):
        """Task at tier 2 → only tries tier 2, 3. Never tier 1."""
        t1_provider = MagicMock()
        t2_provider = MagicMock()
        t2_provider.complete.return_value = "tier 2 ok"

        router = LLMRouter.__new__(LLMRouter)
        router._providers = {}
        router._tier_map = {
            1: [("anthropic", t1_provider)],
            2: [("openai", t2_provider)],
        }
        router._routing = LLMRoutingConfig(chat=2)
        router._escalation = True

        result = router.complete("prompt", task="chat")
        assert result == "tier 2 ok"
        t1_provider.complete.assert_not_called()

    def test_all_tiers_fail_raises_last_error(self):
        """All tiers fail → raise the last error."""
        fail_t1 = MagicMock()
        fail_t1.complete.side_effect = Exception("t1 error")
        fail_t2 = MagicMock()
        fail_t2.complete.side_effect = Exception("t2 error")

        router = LLMRouter.__new__(LLMRouter)
        router._providers = {}
        router._tier_map = {
            1: [("anthropic", fail_t1)],
            2: [("openai", fail_t2)],
        }
        router._routing = LLMRoutingConfig(meta_extraction=1)
        router._escalation = True

        with pytest.raises(Exception, match="t2 error"):
            router.complete("prompt", task="meta_extraction")

    def test_no_provider_at_start_tier_escalates(self):
        """Start tier empty → escalate to next available tier."""
        success_t2 = MagicMock()
        success_t2.complete.return_value = "found at tier 2"

        router = LLMRouter.__new__(LLMRouter)
        router._providers = {}
        router._tier_map = {
            2: [("openai", success_t2)],
        }
        router._routing = LLMRoutingConfig(meta_extraction=1)
        router._escalation = True

        result = router.complete("prompt", task="meta_extraction")
        assert result == "found at tier 2"

    def test_escalation_order_ascending(self):
        """Escalation should try tiers in ascending order: 1 → 2 → 3."""
        call_order = []

        def make_failing(tier_num):
            def side_effect(*a, **kw):
                call_order.append(tier_num)
                raise Exception(f"t{tier_num}")
            return side_effect

        fail_t1 = MagicMock()
        fail_t1.complete.side_effect = make_failing(1)
        fail_t2 = MagicMock()
        fail_t2.complete.side_effect = make_failing(2)
        success_t3 = MagicMock()

        def t3_success(*a, **kw):
            call_order.append(3)
            return "t3 ok"
        success_t3.complete.side_effect = t3_success

        router = LLMRouter.__new__(LLMRouter)
        router._providers = {}
        router._tier_map = {
            1: [("anthropic", fail_t1)],
            3: [("grok", success_t3)],
            2: [("openai", fail_t2)],  # inserted out-of-order to verify sorting
        }
        router._routing = LLMRoutingConfig(meta_extraction=1)
        router._escalation = True

        result = router.complete("prompt", task="meta_extraction")
        assert result == "t3 ok"
        assert call_order == [1, 2, 3]


class TestCreateProvider:
    @patch("pkb.llm.anthropic_provider.anthropic.Anthropic")
    def test_create_provider_grok(self, mock_anthropic_cls):
        """_create_provider should handle 'grok' provider name."""
        from pkb.llm.grok_provider import GrokProvider
        provider = LLMRouter._create_provider("grok", "grok-3-mini-fast", api_key="xai-key")
        assert isinstance(provider, GrokProvider)

    @patch("pkb.llm.anthropic_provider.anthropic.Anthropic")
    def test_create_provider_anthropic_with_api_key(self, mock_cls):
        """_create_provider should pass api_key to AnthropicProvider."""
        from pkb.llm.anthropic_provider import AnthropicProvider
        provider = LLMRouter._create_provider(
            "anthropic", "claude-haiku-4-5-20251001", api_key="sk-test",
        )
        assert isinstance(provider, AnthropicProvider)
        # Verify the API key was forwarded
        mock_cls.assert_called_once_with(api_key="sk-test")

    def test_create_provider_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMRouter._create_provider("unknown", "model")


class TestApiKeyPriority:
    """API key resolution: env var > config.api_key > None (SDK default)."""

    @patch("pkb.llm.anthropic_provider.anthropic.Anthropic")
    def test_env_var_takes_priority_over_config_api_key(
        self, mock_cls, monkeypatch: pytest.MonkeyPatch,
    ):
        """When both env var and config.api_key exist, env var wins."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-123")
        config = LLMConfig(
            providers={
                "anthropic": LLMProviderConfig(
                    api_key_env="ANTHROPIC_API_KEY",
                    api_key="config-key-456",
                    models=[LLMModelEntry(name="claude-haiku-4-5-20251001", tier=1)],
                ),
            },
        )
        LLMRouter(config)
        mock_cls.assert_called_once_with(api_key="env-key-123")

    @patch("pkb.llm.anthropic_provider.anthropic.Anthropic")
    def test_config_api_key_used_when_env_var_missing(
        self, mock_cls, monkeypatch: pytest.MonkeyPatch,
    ):
        """When env var is not set, config.api_key should be used."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = LLMConfig(
            providers={
                "anthropic": LLMProviderConfig(
                    api_key_env="ANTHROPIC_API_KEY",
                    api_key="config-key-456",
                    models=[LLMModelEntry(name="claude-haiku-4-5-20251001", tier=1)],
                ),
            },
        )
        LLMRouter(config)
        mock_cls.assert_called_once_with(api_key="config-key-456")

    @patch("pkb.llm.anthropic_provider.anthropic.Anthropic")
    def test_none_when_neither_env_nor_config(
        self, mock_cls, monkeypatch: pytest.MonkeyPatch,
    ):
        """When no env var and no config.api_key, None passed (SDK reads its own default)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = LLMConfig(
            providers={
                "anthropic": LLMProviderConfig(
                    api_key_env="ANTHROPIC_API_KEY",
                    models=[LLMModelEntry(name="claude-haiku-4-5-20251001", tier=1)],
                ),
            },
        )
        LLMRouter(config)
        mock_cls.assert_called_once_with(api_key=None)

    @patch("pkb.llm.anthropic_provider.anthropic.Anthropic")
    def test_config_api_key_without_api_key_env(
        self, mock_cls,
    ):
        """When api_key_env is not set at all, config.api_key should still work."""
        config = LLMConfig(
            providers={
                "anthropic": LLMProviderConfig(
                    api_key="direct-key-789",
                    models=[LLMModelEntry(name="claude-haiku-4-5-20251001", tier=1)],
                ),
            },
        )
        LLMRouter(config)
        mock_cls.assert_called_once_with(api_key="direct-key-789")


class TestAutoTierResolution:
    """Auto-tier: tier=None models get tier from catalog."""

    def test_explicit_tier_takes_priority(self):
        """Explicit tier=2 should not be overridden by catalog."""
        mock_provider = MagicMock()
        mock_catalog = MagicMock()
        mock_catalog.get_tier.return_value = 1  # catalog says tier 1

        router = LLMRouter.__new__(LLMRouter)
        router._providers = {}
        router._tier_map = {}
        router._routing = LLMRoutingConfig()
        router._escalation = True

        # Simulate what __init__ should do: explicit tier=2 stays 2
        config = LLMConfig(
            providers={
                "anthropic": LLMProviderConfig(
                    models=[LLMModelEntry(name="claude-haiku-4-5-20251001", tier=2)],
                ),
            },
        )
        with patch("pkb.llm.router.LLMRouter._create_provider", return_value=mock_provider):
            router = LLMRouter(config)
        # Should be in tier 2, not tier 1 from catalog
        assert 2 in router._tier_map
        assert len(router._tier_map[2]) == 1

    def test_auto_tier_from_catalog(self):
        """tier=None should resolve from catalog."""
        mock_provider = MagicMock()
        config = LLMConfig(
            providers={
                "anthropic": LLMProviderConfig(
                    models=[LLMModelEntry(name="claude-sonnet-4-6")],  # tier=None
                ),
            },
        )
        with patch("pkb.llm.router.LLMRouter._create_provider", return_value=mock_provider):
            router = LLMRouter(config)
        # claude-sonnet-4-6 is tier 2 in catalog
        assert 2 in router._tier_map
        assert len(router._tier_map[2]) == 1

    def test_auto_tier_unknown_defaults_to_1(self):
        """Unknown model with tier=None should default to tier 1."""
        mock_provider = MagicMock()
        config = LLMConfig(
            providers={
                "anthropic": LLMProviderConfig(
                    models=[LLMModelEntry(name="nonexistent-model-xyz")],  # tier=None
                ),
            },
        )
        with patch("pkb.llm.router.LLMRouter._create_provider", return_value=mock_provider):
            router = LLMRouter(config)
        # Unknown model should default to tier 1
        assert 1 in router._tier_map
        assert len(router._tier_map[1]) == 1

    def test_mixed_explicit_and_auto(self):
        """Mix of explicit and auto-tier models."""
        mock_provider = MagicMock()
        config = LLMConfig(
            providers={
                "anthropic": LLMProviderConfig(
                    models=[
                        LLMModelEntry(name="claude-haiku-4-5-20251001", tier=1),  # explicit
                        LLMModelEntry(name="claude-sonnet-4-6"),  # auto → tier 2
                    ],
                ),
            },
        )
        with patch("pkb.llm.router.LLMRouter._create_provider", return_value=mock_provider):
            router = LLMRouter(config)
        assert 1 in router._tier_map
        assert 2 in router._tier_map


class TestLLMRouterFromConfig:
    def test_from_meta_llm_config(self):
        """Should create router from legacy MetaLLMConfig."""
        from pkb.models.config import MetaLLMConfig

        meta_config = MetaLLMConfig(model="claude-haiku-4-5-20251001")
        router = LLMRouter.from_meta_llm(meta_config)
        assert router is not None
        provider = router.get_provider(task="meta_extraction")
        assert provider is not None


class TestEmptyResponseEscalation:
    """Empty/None/whitespace LLM responses should trigger escalation."""

    def _make_router(self, providers_list, *, escalation=True):
        """Helper to create a router with given provider mocks."""
        router = LLMRouter.__new__(LLMRouter)
        router._providers = {}
        tier_map = {}
        for tier, name, provider in providers_list:
            if tier not in tier_map:
                tier_map[tier] = []
            tier_map[tier].append((name, provider))
        router._tier_map = tier_map
        router._routing = LLMRoutingConfig(meta_extraction=1)
        router._escalation = escalation
        return router

    def test_empty_string_triggers_escalation(self):
        """빈 문자열 응답 → 다음 provider로 escalation."""
        empty_provider = MagicMock()
        empty_provider.complete.return_value = ""
        backup_provider = MagicMock()
        backup_provider.complete.return_value = "good result"

        router = self._make_router([
            (1, "anthropic", empty_provider),
            (1, "openai", backup_provider),
        ])
        result = router.complete("prompt", task="meta_extraction")
        assert result == "good result"

    def test_none_triggers_escalation(self):
        """None 응답 → 다음 provider로 escalation."""
        none_provider = MagicMock()
        none_provider.complete.return_value = None
        backup_provider = MagicMock()
        backup_provider.complete.return_value = "good result"

        router = self._make_router([
            (1, "anthropic", none_provider),
            (1, "openai", backup_provider),
        ])
        result = router.complete("prompt", task="meta_extraction")
        assert result == "good result"

    def test_whitespace_triggers_escalation(self):
        """공백만 있는 응답 → 다음 provider로 escalation."""
        ws_provider = MagicMock()
        ws_provider.complete.return_value = "   \n\t  "
        backup_provider = MagicMock()
        backup_provider.complete.return_value = "good result"

        router = self._make_router([
            (1, "anthropic", ws_provider),
            (1, "openai", backup_provider),
        ])
        result = router.complete("prompt", task="meta_extraction")
        assert result == "good result"

    def test_empty_cross_tier_escalation(self):
        """빈 응답 → tier 1 → tier 2 cross-tier escalation."""
        empty_t1 = MagicMock()
        empty_t1.complete.return_value = ""
        success_t2 = MagicMock()
        success_t2.complete.return_value = "tier 2 ok"

        router = self._make_router([
            (1, "anthropic", empty_t1),
            (2, "openai", success_t2),
        ])
        result = router.complete("prompt", task="meta_extraction")
        assert result == "tier 2 ok"

    def test_all_empty_raises_last_error(self):
        """모든 provider가 빈 응답 → ValueError raise."""
        empty1 = MagicMock()
        empty1.complete.return_value = ""
        empty2 = MagicMock()
        empty2.complete.return_value = None

        router = self._make_router([
            (1, "anthropic", empty1),
            (1, "openai", empty2),
        ])
        with pytest.raises(ValueError, match="Empty response"):
            router.complete("prompt", task="meta_extraction")

    def test_empty_no_escalation_raises(self):
        """escalation=False에서 빈 응답 → 즉시 ValueError."""
        empty_provider = MagicMock()
        empty_provider.complete.return_value = ""

        router = self._make_router([
            (1, "anthropic", empty_provider),
        ], escalation=False)
        with pytest.raises(ValueError, match="Empty response"):
            router.complete("prompt", task="meta_extraction")


class TestRetryLogic:
    """max_retries parameter: retry same provider before escalation."""

    def _make_router(self, providers_list, *, escalation=True):
        router = LLMRouter.__new__(LLMRouter)
        router._providers = {}
        tier_map = {}
        for tier, name, provider in providers_list:
            if tier not in tier_map:
                tier_map[tier] = []
            tier_map[tier].append((name, provider))
        router._tier_map = tier_map
        router._routing = LLMRoutingConfig(meta_extraction=1)
        router._escalation = escalation
        return router

    def test_retry_success_on_second_attempt(self):
        """max_retries=2: 첫 시도 실패 → 두 번째 성공."""
        provider = MagicMock()
        provider.complete.side_effect = [Exception("temp error"), "success"]

        router = self._make_router([(1, "anthropic", provider)])
        result = router.complete("prompt", task="meta_extraction", max_retries=2)
        assert result == "success"
        assert provider.complete.call_count == 2

    def test_retry_exhausted_then_escalation(self):
        """max_retries=2 소진 → 다음 provider로 escalation."""
        failing = MagicMock()
        failing.complete.side_effect = Exception("always fails")
        backup = MagicMock()
        backup.complete.return_value = "backup ok"

        router = self._make_router([
            (1, "anthropic", failing),
            (1, "openai", backup),
        ])
        result = router.complete("prompt", task="meta_extraction", max_retries=2)
        assert result == "backup ok"
        assert failing.complete.call_count == 2

    def test_default_max_retries_is_one(self):
        """기본 max_retries=1: 기존 동작 (retry 없이 바로 escalation)."""
        failing = MagicMock()
        failing.complete.side_effect = Exception("fail")
        backup = MagicMock()
        backup.complete.return_value = "backup"

        router = self._make_router([
            (1, "anthropic", failing),
            (1, "openai", backup),
        ])
        result = router.complete("prompt", task="meta_extraction")
        assert result == "backup"
        assert failing.complete.call_count == 1
