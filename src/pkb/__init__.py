"""PKB — Private Knowledge Base."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pkb")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
