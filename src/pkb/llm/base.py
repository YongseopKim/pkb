"""LLM provider protocol."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0) -> str:
        """Send a prompt and get a text response."""
        ...

    def model_name(self) -> str:
        """Return the model name."""
        ...
