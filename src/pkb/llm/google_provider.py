"""Google GenAI LLM provider."""

from google import genai
from google.genai import types


class GoogleProvider:
    """LLM provider using Google GenAI API."""

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None) -> None:
        self._model_name = model
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0) -> str:
        """Send a prompt to Google GenAI and return text response."""
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config,
        )
        return response.text

    def model_name(self) -> str:
        return self._model_name
