"""PKB home initialization."""

import shutil
from pathlib import Path

from pkb.config import create_default_config, get_pkb_home
from pkb.constants import (
    CONFIG_FILENAME,
    DOMAINS_FILENAME,
    INDEX_DIR,
    TOPICS_FILENAME,
    VOCAB_DIR,
)

_DATA_DIR = Path(__file__).resolve().parent / "data"


def init_pkb_home(*, force: bool = False) -> Path:
    """Initialize ~/.pkb/ directory structure.

    Creates:
        ~/.pkb/
        ├── config.yaml
        ├── vocab/
        │   ├── domains.yaml
        │   └── topics.yaml
        └── index/

    Args:
        force: If True, overwrite existing files.

    Returns:
        Path to the PKB home directory.

    Raises:
        FileExistsError: If PKB home already has a config and force is False.
    """
    pkb_home = get_pkb_home()
    config_path = pkb_home / CONFIG_FILENAME

    if config_path.exists() and not force:
        raise FileExistsError(
            f"PKB home already initialized at {pkb_home}. "
            "Use --force to overwrite."
        )

    # Create directory structure
    pkb_home.mkdir(parents=True, exist_ok=True)
    (pkb_home / VOCAB_DIR).mkdir(exist_ok=True)
    (pkb_home / INDEX_DIR).mkdir(exist_ok=True)

    # Create default config
    create_default_config(config_path)

    # Copy bundled vocab files
    shutil.copy2(_DATA_DIR / DOMAINS_FILENAME, pkb_home / VOCAB_DIR / DOMAINS_FILENAME)
    shutil.copy2(_DATA_DIR / TOPICS_FILENAME, pkb_home / VOCAB_DIR / TOPICS_FILENAME)

    return pkb_home
