"""Tests for LLM providers."""

from unittest.mock import MagicMock, patch

from pkb.llm.anthropic_provider import AnthropicProvider
from pkb.llm.google_provider import GoogleProvider
from pkb.llm.grok_provider import GrokProvider
from pkb.llm.openai_provider import OpenAIProvider


class TestAnthropicProvider:
    @patch("pkb.llm.anthropic_provider.anthropic.Anthropic")
    def test_complete(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="Hello world")]
        mock_client.messages.create.return_value = mock_msg

        provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
        result = provider.complete("Say hello", max_tokens=100)
        assert result == "Hello world"

    @patch("pkb.llm.anthropic_provider.anthropic.Anthropic")
    def test_complete_with_temperature(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="Result")]
        mock_client.messages.create.return_value = mock_msg

        provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
        provider.complete("Prompt", max_tokens=100, temperature=0.7)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    @patch("pkb.llm.anthropic_provider.anthropic.Anthropic")
    def test_passes_api_key(self, mock_cls):
        """api_key should be forwarded to Anthropic client."""
        AnthropicProvider(model="claude-haiku-4-5-20251001", api_key="sk-test-123")
        mock_cls.assert_called_once_with(api_key="sk-test-123")

    @patch("pkb.llm.anthropic_provider.anthropic.Anthropic")
    def test_api_key_none_by_default(self, mock_cls):
        """When api_key is not given, None should be passed (SDK uses env var)."""
        AnthropicProvider()
        mock_cls.assert_called_once_with(api_key=None)

    def test_implements_protocol(self):
        """AnthropicProvider should satisfy LLMProvider protocol."""
        assert hasattr(AnthropicProvider, "complete")
        assert hasattr(AnthropicProvider, "model_name")


class TestOpenAIProvider:
    @patch("pkb.llm.openai_provider.openai.OpenAI")
    def test_complete(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "OpenAI response"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_resp

        provider = OpenAIProvider(model="gpt-4o-mini")
        result = provider.complete("Say hello", max_tokens=100)
        assert result == "OpenAI response"

    def test_implements_protocol(self):
        assert hasattr(OpenAIProvider, "complete")
        assert hasattr(OpenAIProvider, "model_name")


class TestGoogleProvider:
    @patch("pkb.llm.google_provider.genai.Client")
    def test_complete(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Google response"
        mock_client.models.generate_content.return_value = mock_response

        provider = GoogleProvider(model="gemini-2.0-flash")
        result = provider.complete("Say hello", max_tokens=100)
        assert result == "Google response"

    def test_implements_protocol(self):
        assert hasattr(GoogleProvider, "complete")
        assert hasattr(GoogleProvider, "model_name")


class TestGrokProvider:
    @patch("pkb.llm.grok_provider.openai.OpenAI")
    def test_complete(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "Grok response"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_resp

        provider = GrokProvider(model="grok-3-mini-fast")
        result = provider.complete("Say hello", max_tokens=100)
        assert result == "Grok response"

    @patch("pkb.llm.grok_provider.openai.OpenAI")
    def test_uses_xai_base_url(self, mock_cls):
        """GrokProvider must set base_url to xAI API."""
        GrokProvider()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == "https://api.x.ai/v1"

    @patch("pkb.llm.grok_provider.openai.OpenAI")
    def test_passes_api_key(self, mock_cls):
        GrokProvider(api_key="xai-test-key")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["api_key"] == "xai-test-key"

    def test_implements_protocol(self):
        assert hasattr(GrokProvider, "complete")
        assert hasattr(GrokProvider, "model_name")
