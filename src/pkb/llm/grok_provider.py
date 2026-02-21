"""Grok (xAI) LLM provider — OpenAI-compatible API."""

import openai


class GrokProvider:
    """LLM provider using xAI Grok API (OpenAI SDK compatible)."""

    def __init__(self, model: str = "grok-3-mini-fast", api_key: str | None = None) -> None:
        self._model = model
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0) -> str:
        """Send a prompt to Grok and return text response."""
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def model_name(self) -> str:
        return self._model
