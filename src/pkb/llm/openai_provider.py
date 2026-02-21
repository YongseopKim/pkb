"""OpenAI LLM provider."""

import openai


class OpenAIProvider:
    """LLM provider using OpenAI API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self._model = model
        self._client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        self._skip_temperature = False  # auto-detected for reasoning models

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0) -> str:
        """Send a prompt to OpenAI and return text response."""
        kwargs: dict = {
            "model": self._model,
            "max_completion_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not self._skip_temperature:
            kwargs["temperature"] = temperature

        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            if "temperature" in str(e) and not self._skip_temperature:
                self._skip_temperature = True
                kwargs.pop("temperature", None)
                response = self._client.chat.completions.create(**kwargs)
            else:
                raise
        return response.choices[0].message.content

    def model_name(self) -> str:
        return self._model
