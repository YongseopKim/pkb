"""Shared test fixtures."""

from pathlib import Path

import pytest

# Root of the main repo (not the worktree) for sample data
REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "exporter-examples" / "jsonl" / "PKB"
MD_SAMPLES_DIR = REPO_ROOT / "exporter-examples" / "md" / "PKB"


@pytest.fixture
def samples_dir() -> Path:
    """Path to JSONL sample files."""
    return SAMPLES_DIR


@pytest.fixture
def md_samples_dir() -> Path:
    """Path to MD sample files."""
    return MD_SAMPLES_DIR


@pytest.fixture
def tmp_pkb_home(tmp_path: Path) -> Path:
    """Temporary PKB home directory for test isolation."""
    pkb_home = tmp_path / ".pkb"
    pkb_home.mkdir()
    return pkb_home
