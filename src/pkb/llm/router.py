"""LLM Router — tier-based routing with escalation."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import TYPE_CHECKING

from pkb.llm.base import LLMProvider
from pkb.models.config import LLMConfig, MetaLLMConfig

if TYPE_CHECKING:
    pass


class LLMRouter:
    """Routes LLM calls to the appropriate provider based on task tier."""

    def __init__(self, config: LLMConfig) -> None:
        self._routing = config.routing
        self._escalation = config.routing.escalation
        self._providers: dict[str, list[LLMProvider]] = {}
        self._tier_map: dict[int, list[tuple[str, LLMProvider]]] = defaultdict(list)

        # Lazy-load catalog only when needed (tier=None models exist).
        catalog = None

        for provider_name, provider_config in config.providers.items():
            api_key = None
            if provider_config.api_key_env:
                api_key = os.environ.get(provider_config.api_key_env)
            if not api_key and provider_config.api_key:
                api_key = provider_config.api_key

            for model_entry in provider_config.models:
                provider = self._create_provider(
                    provider_name, model_entry.name, api_key,
                )
                if provider_name not in self._providers:
                    self._providers[provider_name] = []
                self._providers[provider_name].append(provider)

                # Resolve tier: explicit > catalog > default 1
                tier = model_entry.tier
                if tier is None:
                    if catalog is None:
                        from pkb.llm.catalog import load_model_catalog
                        catalog = load_model_catalog()
                    tier = catalog.get_tier(provider_name, model_entry.name) or 1

                self._tier_map[tier].append((provider_name, provider))

    @staticmethod
    def _create_provider(
        provider_name: str, model: str, api_key: str | None = None,
    ) -> LLMProvider:
        """Create a provider instance."""
        if provider_name == "anthropic":
            from pkb.llm.anthropic_provider import AnthropicProvider
            return AnthropicProvider(model=model, api_key=api_key)
        elif provider_name == "openai":
            from pkb.llm.openai_provider import OpenAIProvider
            return OpenAIProvider(model=model, api_key=api_key)
        elif provider_name == "google":
            from pkb.llm.google_provider import GoogleProvider
            return GoogleProvider(model=model, api_key=api_key)
        elif provider_name == "grok":
            from pkb.llm.grok_provider import GrokProvider
            return GrokProvider(model=model, api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    @classmethod
    def from_meta_llm(cls, config: MetaLLMConfig) -> LLMRouter:
        """Create a router from legacy MetaLLMConfig (backward compatibility).

        Sets all task tiers to 1 since legacy config only registers one model at tier 1.
        """
        from pkb.models.config import LLMModelEntry, LLMProviderConfig, LLMRoutingConfig

        llm_config = LLMConfig(
            default_provider=config.provider,
            providers={
                config.provider: LLMProviderConfig(
                    models=[LLMModelEntry(name=config.model, tier=1)],
                ),
            },
            routing=LLMRoutingConfig(meta_extraction=1, chat=1, escalation=True),
        )
        return cls(llm_config)

    def get_provider(self, task: str = "meta_extraction") -> LLMProvider:
        """Get the provider for a given task type."""
        tier = getattr(self._routing, task, 1)
        providers = self._tier_map.get(tier, [])
        if not providers:
            raise ValueError(
                f"No provider configured for tier {tier} (task: {task}). "
                f"Available tiers: {list(self._tier_map.keys())}"
            )
        return providers[0][1]

    def complete(
        self,
        prompt: str,
        *,
        task: str = "meta_extraction",
        max_tokens: int = 1024,
        temperature: float = 0,
        max_retries: int = 1,
    ) -> str:
        """Route a completion call to the appropriate provider.

        With escalation enabled, tries all providers in the start tier first,
        then escalates to higher tiers in ascending order (cross-tier escalation).

        Each provider is retried up to max_retries times before moving to the next.
        Empty/None/whitespace responses are treated as errors and trigger retry/escalation.
        """
        start_tier = getattr(self._routing, task, 1)

        if self._escalation:
            tiers_to_try = sorted(t for t in self._tier_map if t >= start_tier)
        else:
            tiers_to_try = [start_tier]

        last_error = None
        for tier in tiers_to_try:
            providers = self._tier_map.get(tier, [])
            for _name, provider in providers:
                for _attempt in range(max_retries):
                    try:
                        result = provider.complete(
                            prompt, max_tokens=max_tokens, temperature=temperature,
                        )
                        if not result or not result.strip():
                            raise ValueError(f"Empty response from {_name}")
                        return result
                    except Exception as e:
                        last_error = e
                        if not self._escalation:
                            raise
                        continue

        if last_error is not None:
            raise last_error
        raise ValueError(f"No provider configured for tier {start_tier} (task: {task})")
