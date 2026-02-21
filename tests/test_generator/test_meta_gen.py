"""Tests for meta generator (Router-based, mock-based)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from pkb.generator.meta_gen import MetaGenerator
from pkb.models.config import MetaLLMConfig
from pkb.models.meta import BundleMeta, ResponseMeta


@pytest.fixture
def mock_router():
    """Mock LLMRouter."""
    router = MagicMock()
    router.complete.return_value = "{}"
    return router


@pytest.fixture
def gen(mock_router):
    config = MetaLLMConfig()
    return MetaGenerator(config, router=mock_router)


class TestParseJsonResponse:
    def test_plain_json(self, gen):
        text = '{"summary": "테스트", "key_claims": [], "stance": "neutral"}'
        result = gen._parse_json_response(text)
        assert result["summary"] == "테스트"

    def test_json_in_markdown_fence(self, gen):
        text = '```json\n{"summary": "테스트"}\n```'
        result = gen._parse_json_response(text)
        assert result["summary"] == "테스트"

    def test_json_in_plain_fence(self, gen):
        text = '```\n{"summary": "테스트"}\n```'
        result = gen._parse_json_response(text)
        assert result["summary"] == "테스트"

    def test_invalid_json_raises(self, gen):
        with pytest.raises(ValueError):
            gen._parse_json_response("not json at all")


class TestGenerateResponseMeta:
    def test_returns_response_meta(self, gen, mock_router):
        mock_router.complete.return_value = json.dumps({
            "summary": "Python 비동기 설명",
            "key_claims": ["asyncio 기반"],
            "stance": "informative",
            "model": "claude-sonnet",
        })
        result = gen.generate_response_meta(
            platform="claude",
            content="async 설명 내용...",
        )
        assert isinstance(result, ResponseMeta)
        assert result.platform == "claude"
        assert result.summary == "Python 비동기 설명"

    def test_router_called_with_meta_extraction_task(self, gen, mock_router):
        mock_router.complete.return_value = json.dumps({
            "summary": "요약",
            "key_claims": [],
            "stance": "neutral",
        })
        gen.generate_response_meta(platform="claude", content="test")
        call_kwargs = mock_router.complete.call_args[1]
        assert call_kwargs["task"] == "meta_extraction"
        assert call_kwargs["temperature"] == 0


class TestGenerateBundleMeta:
    def test_returns_bundle_meta(self, gen, mock_router):
        mock_router.complete.return_value = json.dumps({
            "summary": "PKB 설계 토론",
            "slug": "pkb-system-design",
            "domains": ["dev"],
            "topics": ["system-design"],
            "pending_topics": [],
            "consensus": "전원 합의",
            "divergence": "의견 차이",
        })
        result = gen.generate_bundle_meta(
            question="PKB 설계에 대해",
            platforms=["claude", "chatgpt"],
            response_summaries="Claude: ...\nChatGPT: ...",
            available_domains=["dev", "invest"],
            available_topics=["system-design", "python"],
        )
        assert isinstance(result, BundleMeta)
        assert result.slug == "pkb-system-design"
        assert "dev" in result.domains


class TestMetaGeneratorAutoRouter:
    @patch("pkb.llm.router.LLMRouter.from_meta_llm")
    def test_auto_builds_router_when_none(self, mock_from_meta):
        """When router=None, MetaGenerator should auto-build one via from_meta_llm."""
        mock_router = MagicMock()
        mock_from_meta.return_value = mock_router
        config = MetaLLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
        gen = MetaGenerator(config, router=None)
        assert gen._router is mock_router
        mock_from_meta.assert_called_once_with(config)
