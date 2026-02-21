"""Anthropic LLM provider."""

import anthropic


class AnthropicProvider:
    """LLM provider using Anthropic API."""

    def __init__(
        self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None,
    ) -> None:
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0) -> str:
        """Send a prompt to Anthropic and return text response."""
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def model_name(self) -> str:
        return self._model
