"""Chat engine — RAG pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pkb.chat.context import assemble_context
from pkb.chat.models import ChatResponse, ChatSession
from pkb.search.models import SearchMode, SearchQuery

if TYPE_CHECKING:
    from pkb.llm.router import LLMRouter
    from pkb.search.engine import SearchEngine


class ChatEngine:
    """RAG chatbot engine: question -> search -> context -> LLM -> response."""

    def __init__(
        self,
        *,
        search_engine: SearchEngine,
        router: LLMRouter,
        kb: str | None = None,
        max_results: int = 5,
        max_history: int = 10,
    ) -> None:
        self._search = search_engine
        self._router = router
        self._kb = kb
        self._max_results = max_results
        self._max_history = max_history

    def ask(self, question: str, *, session: ChatSession) -> ChatResponse:
        """Process a user question through the RAG pipeline.

        1. Search knowledge base for relevant bundles
        2. Assemble context with search results + conversation history
        3. Call LLM to generate answer
        4. Extract sources and return response
        """
        # Step 1: Search
        search_query = SearchQuery(
            query=question,
            mode=SearchMode.HYBRID,
            kb=self._kb,
            limit=self._max_results,
        )
        results = self._search.search(search_query)

        # Step 2: Assemble context
        history = session.get_history(max_messages=self._max_history)
        prompt = assemble_context(
            question=question,
            search_results=results,
            history=history,
            max_results=self._max_results,
        )

        # Step 3: Call LLM
        raw_response = self._router.complete(
            prompt,
            task="chat",
            max_tokens=2048,
            temperature=0.3,
        )

        # Step 4: Extract sources
        sources = [
            {"bundle_id": r.bundle_id, "summary": r.summary}
            for r in results
        ]

        response = ChatResponse(content=raw_response, sources=sources)

        # Update session
        session.add_message("user", question)
        session.add_message("assistant", raw_response)

        return response
