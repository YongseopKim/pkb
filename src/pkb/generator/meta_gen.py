"""Meta generation via LLM API (always uses LLMRouter)."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from pkb.generator.prompts import load_prompt, render_prompt
from pkb.models.config import MetaLLMConfig
from pkb.models.meta import BundleMeta, ResponseMeta

if TYPE_CHECKING:
    from pkb.llm.router import LLMRouter


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
        raw = self._call_api(prompt)
        data = self._parse_json_response(raw)
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
        raw = self._call_api(prompt)
        data = self._parse_json_response(raw)
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

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences."""
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {e}\nRaw: {text}")
