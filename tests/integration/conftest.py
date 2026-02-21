"""Fixtures for LLM integration tests using real ~/.pkb/config.yaml."""

import os

import pytest

SKIP_ALL = not os.environ.get("PKB_LLM_INTEGRATION")


@pytest.fixture
def real_config():
    """Load the real PKB config from ~/.pkb/config.yaml."""
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME

    pkb_home = get_pkb_home()
    config_path = pkb_home / CONFIG_FILENAME
    if not config_path.exists():
        pytest.skip(f"Config not found: {config_path}")

    return load_config(config_path)


@pytest.fixture
def real_router(real_config):
    """Build a real LLMRouter from the loaded config."""
    from pkb.config import build_llm_router

    return build_llm_router(real_config)
