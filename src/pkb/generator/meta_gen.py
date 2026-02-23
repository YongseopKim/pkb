"""Meta generation via LLM API (always uses LLMRouter)."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from pkb.generator.prompts import load_prompt, render_prompt
from pkb.models.config import MetaLLMConfig
from pkb.models.meta import BundleMeta, ResponseMeta

if TYPE_CHECKING:
    from pkb.llm.router import LLMRouter

logger = logging.getLogger(__name__)


class MetaGenerator:
    """Generates metadata using LLM API via LLMRouter."""

    def __init__(
        self,
        config: MetaLLMConfig,
        router: LLMRouter | None = None,
    ) -> None:
        self._config = config
        if router is None:
            from pkb.llm.router import LLMRouter as _LLMRouter
            router = _LLMRouter.from_meta_llm(config)
        self._router = router

    def generate_response_meta(
        self, *, platform: str, content: str
    ) -> ResponseMeta:
        """Generate metadata for a single LLM response."""
        template = load_prompt("response_meta")
        prompt = render_prompt(template, platform=platform, content=content)
        data = self._call_api_with_json_retry(prompt)
        data["platform"] = platform
        return ResponseMeta(**data)

    def generate_bundle_meta(
        self,
        *,
        question: str,
        platforms: list[str],
        response_summaries: str,
        available_domains: list[str],
        available_topics: list[str],
    ) -> BundleMeta:
        """Generate aggregate metadata for a bundle."""
        template = load_prompt("bundle_meta")
        prompt = render_prompt(
            template,
            question=question,
            platforms=", ".join(platforms),
            domains=", ".join(available_domains),
            topics=", ".join(available_topics),
            response_summaries=response_summaries,
        )
        data = self._call_api_with_json_retry(prompt)
        return BundleMeta(**data)

    def _call_api(self, prompt: str) -> str:
        """Call LLM API via Router."""
        return self._router.complete(
            prompt,
            task="meta_extraction",
            max_tokens=1024,
            temperature=self._config.temperature,
            max_retries=self._config.max_retries,
        )

    def _call_api_with_json_retry(self, prompt: str) -> dict:
        """Call LLM API and parse JSON, with one retry on parse failure."""
        raw = self._call_api(prompt)
        try:
            return self._parse_json_response(raw)
        except ValueError:
            logger.warning("JSON parse failed, retrying with correction prompt")
            retry_prompt = (
                "Your previous response was not valid JSON. "
                "Return ONLY a JSON object with the required fields. "
                "No explanations, no markdown, just the JSON object.\n\n"
                f"Original prompt:\n{prompt}"
            )
            raw2 = self._call_api(retry_prompt)
            return self._parse_json_response(raw2)

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences."""
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            text = self._normalize_invalid_escapes(text)
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSON from LLM response: {e}\nRaw: {text}"
                )

    _INVALID_ESCAPE_RE = re.compile(r'\\([^"\\/bfnrtu])')

    @staticmethod
    def _normalize_invalid_escapes(text: str) -> str:
        """Remove backslash from invalid JSON escape sequences (e.g. \\$ → $)."""
        return MetaGenerator._INVALID_ESCAPE_RE.sub(r'\1', text)
