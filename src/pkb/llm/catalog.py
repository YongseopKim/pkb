"""LLM Model Catalog — tier and pricing lookup from bundled YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Module-level cache to avoid re-loading on every call.
_catalog_cache: ModelCatalog | None = None


class ModelInfo(BaseModel):
    """Pricing and tier metadata for a single model."""

    tier: int
    input_price: float
    output_price: float
    context_window: int
    note: str | None = None


class ModelCatalog:
    """Lookup table: (provider, model_name) → ModelInfo."""

    def __init__(self, data: dict[str, dict[str, ModelInfo]]) -> None:
        self._data = data  # {provider: {model_name: ModelInfo}}

    def get_tier(self, provider: str, model: str) -> int | None:
        """Return the tier for a model, or None if unknown."""
        info = self.get_info(provider, model)
        return info.tier if info else None

    def get_info(self, provider: str, model: str) -> ModelInfo | None:
        """Return full ModelInfo for a model, or None if unknown."""
        return self._data.get(provider, {}).get(model)

    def providers(self) -> list[str]:
        """Return list of provider names in the catalog."""
        return list(self._data.keys())


def load_model_catalog(path: Path | None = None) -> ModelCatalog:
    """Load model catalog from YAML.

    Uses bundled data/model_catalog.yaml by default.
    Results are cached at module level (only for the default bundled path).
    """
    global _catalog_cache

    if path is None:
        if _catalog_cache is not None:
            return _catalog_cache
        path = _DATA_DIR / "model_catalog.yaml"

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    data: dict[str, dict[str, ModelInfo]] = {}
    for provider_name, provider_data in raw.get("providers", {}).items():
        models: dict[str, ModelInfo] = {}
        for model_name, model_data in provider_data.get("models", {}).items():
            models[model_name] = ModelInfo(**model_data)
        data[provider_name] = models

    catalog = ModelCatalog(data)

    # Cache only for the bundled default path.
    if path == _DATA_DIR / "model_catalog.yaml":
        _catalog_cache = catalog

    return catalog
