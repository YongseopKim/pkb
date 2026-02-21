"""Tests for LLM Model Catalog."""

from pathlib import Path

import yaml

from pkb.llm.catalog import ModelCatalog, ModelInfo, load_model_catalog


class TestLoadBundledCatalog:
    def test_load_bundled_catalog(self):
        """Should load the bundled model_catalog.yaml successfully."""
        catalog = load_model_catalog()
        assert isinstance(catalog, ModelCatalog)

    def test_catalog_all_providers(self):
        """Bundled catalog should contain all 4 providers."""
        catalog = load_model_catalog()
        providers = catalog.providers()
        for name in ("anthropic", "openai", "google", "grok"):
            assert name in providers, f"Missing provider: {name}"


class TestGetTier:
    def test_get_tier_known_model(self):
        """Known model should return its tier."""
        catalog = load_model_catalog()
        tier = catalog.get_tier("anthropic", "claude-haiku-4-5-20251001")
        assert tier == 1

    def test_get_tier_unknown_model(self):
        """Unknown model should return None."""
        catalog = load_model_catalog()
        tier = catalog.get_tier("anthropic", "nonexistent-model")
        assert tier is None

    def test_catalog_tier_values(self):
        """Spot-check tier assignments across providers."""
        catalog = load_model_catalog()
        # Tier 1
        assert catalog.get_tier("anthropic", "claude-haiku-4-5-20251001") == 1
        assert catalog.get_tier("openai", "gpt-4o-mini") == 1
        assert catalog.get_tier("google", "gemini-2.0-flash") == 1
        # Tier 2
        assert catalog.get_tier("anthropic", "claude-sonnet-4-6") == 2
        assert catalog.get_tier("openai", "gpt-4o") == 2
        # Tier 3
        assert catalog.get_tier("anthropic", "claude-opus-4-6") == 3


class TestGetInfo:
    def test_get_info_known_model(self):
        """Known model should return full ModelInfo."""
        catalog = load_model_catalog()
        info = catalog.get_info("anthropic", "claude-haiku-4-5-20251001")
        assert isinstance(info, ModelInfo)
        assert info.tier == 1
        assert info.input_price == 1.00
        assert info.output_price == 5.00
        assert info.context_window == 200_000

    def test_get_info_unknown_returns_none(self):
        """Unknown model should return None."""
        catalog = load_model_catalog()
        info = catalog.get_info("openai", "nonexistent-model")
        assert info is None

    def test_model_info_note_field(self):
        """Models with notes should have the note field populated."""
        catalog = load_model_catalog()
        info = catalog.get_info("google", "gemini-2.5-flash")
        assert info is not None
        assert info.note is not None
        assert "thinking" in info.note


class TestModelInfoValidation:
    def test_model_info_fields(self):
        """ModelInfo should have all expected fields."""
        info = ModelInfo(tier=1, input_price=0.5, output_price=1.0, context_window=128000)
        assert info.tier == 1
        assert info.input_price == 0.5
        assert info.output_price == 1.0
        assert info.context_window == 128000
        assert info.note is None

    def test_model_info_with_note(self):
        """ModelInfo should accept optional note."""
        info = ModelInfo(
            tier=2, input_price=3.0, output_price=15.0,
            context_window=200000, note="extended thinking",
        )
        assert info.note == "extended thinking"


class TestCustomPath:
    def test_load_custom_path(self, tmp_path: Path):
        """Should load catalog from a custom YAML path."""
        custom_catalog = {
            "providers": {
                "test_provider": {
                    "models": {
                        "test-model-v1": {
                            "tier": 1,
                            "input_price": 0.1,
                            "output_price": 0.4,
                            "context_window": 100000,
                        }
                    }
                }
            }
        }
        path = tmp_path / "custom_catalog.yaml"
        path.write_text(yaml.dump(custom_catalog))

        catalog = load_model_catalog(path=path)
        assert catalog.get_tier("test_provider", "test-model-v1") == 1
        info = catalog.get_info("test_provider", "test-model-v1")
        assert info is not None
        assert info.input_price == 0.1
