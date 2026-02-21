"""Project-wide constants."""

from pathlib import Path

# Supported platforms from llm-chat-exporter
PLATFORMS = frozenset({"chatgpt", "claude", "gemini", "grok", "perplexity"})

# Default PKB home directory
DEFAULT_PKB_HOME = Path("~/.pkb").expanduser()

# Subdirectories inside PKB home
VOCAB_DIR = "vocab"
INDEX_DIR = "index"

# Config filename
CONFIG_FILENAME = "config.yaml"

# Vocab filenames
DOMAINS_FILENAME = "domains.yaml"
TOPICS_FILENAME = "topics.yaml"

# Inbox cleanup
DONE_DIR_NAME = ".done"
