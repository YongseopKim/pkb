"""Context assembler for RAG prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pkb.generator.prompts import load_prompt

if TYPE_CHECKING:
    from pkb.chat.models import ChatMessage
    from pkb.search.models import BundleSearchResult


def assemble_context(
    *,
    question: str,
    search_results: list[BundleSearchResult],
    history: list[ChatMessage],
    max_results: int = 5,
) -> str:
    """Assemble an LLM prompt from search results and conversation history.

    Budget (approximate tokens):
    - System prompt: ~500
    - Search results: ~3000 (top 5 bundles)
    - History: ~2000 (sliding window)
    """
    system_prompt = _load_system_prompt()

    # Build context from search results
    context_parts = []
    sources = search_results[:max_results]
    for i, result in enumerate(sources, 1):
        context_parts.append(
            f"[{i}] Bundle: {result.bundle_id}\n"
            f"    Summary: {result.summary or '(none)'}\n"
            f"    Domains: {', '.join(result.domains)}\n"
            f"    Score: {result.score:.2f}"
        )

    context_block = "\n\n".join(context_parts) if context_parts else "(검색 결과 없음)"

    # Build conversation history
    history_parts = []
    for msg in history:
        history_parts.append(f"{msg.role}: {msg.content}")
    history_block = "\n".join(history_parts) if history_parts else "(첫 질문)"

    # Assemble full prompt
    prompt = (
        f"{system_prompt}\n\n"
        f"## 검색된 지식베이스 컨텍스트\n\n{context_block}\n\n"
        f"## 대화 이력\n\n{history_block}\n\n"
        f"## 현재 질문\n\n{question}\n\n"
        f"위 컨텍스트를 참고하여 질문에 답변하세요. "
        f"답변 끝에 참조한 출처를 '[번호] bundle_id' 형태로 표시하세요."
    )

    return prompt


def _load_system_prompt() -> str:
    """Load the chat system prompt."""
    try:
        return load_prompt("chat_system")
    except FileNotFoundError:
        return (
            "당신은 PKB(Private Knowledge Base)의 RAG 챗봇입니다. "
            "사용자의 지식베이스에 저장된 대화 내용을 바탕으로 질문에 답변합니다. "
            "답변은 한국어로 제공하고, 출처를 명확히 밝혀주세요."
        )
