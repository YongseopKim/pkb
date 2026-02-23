"""Smart Digest — topic/domain knowledge summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pkb.models.config import DigestConfig
from pkb.search.models import SearchMode, SearchQuery

if TYPE_CHECKING:
    from pkb.db.postgres import BundleRepository
    from pkb.llm.router import LLMRouter
    from pkb.search.engine import SearchEngine


@dataclass
class DigestResult:
    """Result of a digest operation."""

    content: str
    sources: list[dict] = field(default_factory=list)
    bundle_count: int = 0
    topic: str | None = None
    domain: str | None = None


class DigestEngine:
    """Generates topic/domain knowledge summaries from the knowledge base."""

    def __init__(
        self,
        *,
        repo: BundleRepository,
        search_engine: SearchEngine,
        router: LLMRouter,
        config: DigestConfig,
    ) -> None:
        self._repo = repo
        self._search = search_engine
        self._router = router
        self._max_bundles = config.max_bundles
        self._max_tokens = config.max_tokens

    def digest_topic(
        self,
        topic: str,
        *,
        kb: str | None = None,
    ) -> DigestResult:
        """Generate a comprehensive digest for a topic."""
        query = SearchQuery(
            query=topic,
            mode=SearchMode.HYBRID,
            kb=kb,
            topics=[topic],
            limit=self._max_bundles,
        )
        results = self._search.search(query)

        if not results:
            return DigestResult(
                content=f"'{topic}' 토픽에 관련된 번들을 찾을 수 없습니다.",
                topic=topic,
            )

        context = self._build_digest_context(results)
        prompt = self._build_digest_prompt(
            subject=f"토픽: {topic}",
            context=context,
        )

        summary = self._router.complete(
            prompt,
            task="chat",
            max_tokens=self._max_tokens,
            temperature=0.3,
        )

        sources = [
            {"bundle_id": r.bundle_id, "summary": r.summary}
            for r in results
        ]

        return DigestResult(
            content=summary,
            sources=sources,
            bundle_count=len(results),
            topic=topic,
        )

    def digest_domain(
        self,
        domain: str,
        *,
        kb: str | None = None,
    ) -> DigestResult:
        """Generate a comprehensive digest for a domain."""
        bundles = self._repo.list_bundles_by_domain(domain, kb=kb)

        if not bundles:
            return DigestResult(
                content=f"'{domain}' 도메인에 관련된 번들을 찾을 수 없습니다.",
                domain=domain,
            )

        context_parts = []
        for i, b in enumerate(bundles[: self._max_bundles], 1):
            context_parts.append(
                f"[{i}] {b['bundle_id']}: {b['question']}\n"
                f"    요약: {b.get('summary', '(없음)')}"
            )
        context = "\n\n".join(context_parts)

        prompt = self._build_digest_prompt(
            subject=f"도메인: {domain}",
            context=context,
        )

        summary = self._router.complete(
            prompt,
            task="chat",
            max_tokens=self._max_tokens,
            temperature=0.3,
        )

        sources = [
            {"bundle_id": b["bundle_id"], "summary": b.get("summary")}
            for b in bundles[: self._max_bundles]
        ]

        return DigestResult(
            content=summary,
            sources=sources,
            bundle_count=len(bundles),
            domain=domain,
        )

    def _build_digest_context(self, results: list) -> str:
        """Build context string from search results."""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] {r.bundle_id}: {r.question}\n"
                f"    요약: {r.summary or '(없음)'}\n"
                f"    도메인: {', '.join(r.domains)}\n"
                f"    토픽: {', '.join(r.topics)}"
            )
        return "\n\n".join(parts)

    def _build_digest_prompt(self, *, subject: str, context: str) -> str:
        """Build the LLM prompt for digest generation."""
        return (
            "당신은 PKB(Private Knowledge Base)의 지식 분석가입니다.\n\n"
            f"## 분석 대상: {subject}\n\n"
            f"## 관련 번들 목록\n\n{context}\n\n"
            "## 지시사항\n\n"
            "위 번들들을 종합하여 다음을 포함하는 한국어 리포트를 작성하세요:\n"
            "1. **핵심 요약**: 이 주제에 대해 축적된 지식의 전체 요약\n"
            "2. **주요 인사이트**: 반복적으로 등장하는 핵심 개념이나 결론\n"
            "3. **관점 변화**: 시간에 따른 관점이나 이해의 변화 (있다면)\n"
            "4. **출처**: 참조한 번들 ID 목록\n"
        )
