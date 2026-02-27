"""Embedding abstraction layer for PKB."""

from pkb.embedding.base import Embedder
from pkb.embedding.factory import create_embedder

__all__ = ["Embedder", "create_embedder"]
