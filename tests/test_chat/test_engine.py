"""Tests for ChatEngine."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from pkb.chat.engine import ChatEngine
from pkb.chat.models import ChatSession
from pkb.search.models import BundleSearchResult


@pytest.fixture
def mock_search_engine():
    engine = MagicMock()
    engine.search.return_value = [
        BundleSearchResult(
            bundle_id="20260101-test-abc1",
            question="Python async 패턴은?",
            summary="Python 비동기 프로그래밍 패턴 설명",
            domains=["dev"],
            topics=["python"],
            score=0.85,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            source="both",
        ),
    ]
    return engine


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.complete.return_value = (
        "Python의 async/await는 코루틴 기반입니다.\n\n"
        "출처:\n[1] 20260101-test-abc1 — Python async 패턴은?"
    )
    return router


@pytest.fixture
def chat_engine(mock_search_engine, mock_router):
    return ChatEngine(
        search_engine=mock_search_engine,
        router=mock_router,
    )


class TestChatEngine:
    def test_ask_returns_response(self, chat_engine):
        session = ChatSession()
        response = chat_engine.ask("Python async 패턴이 뭐야?", session=session)
        assert response.content is not None
        assert len(response.content) > 0

    def test_ask_searches_bundles(self, chat_engine, mock_search_engine):
        session = ChatSession()
        chat_engine.ask("Python async", session=session)
        mock_search_engine.search.assert_called_once()

    def test_ask_calls_llm(self, chat_engine, mock_router):
        session = ChatSession()
        chat_engine.ask("Python async", session=session)
        mock_router.complete.assert_called_once()

    def test_ask_updates_session(self, chat_engine):
        session = ChatSession()
        chat_engine.ask("Python async", session=session)
        # Should have user message + assistant response
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"

    def test_ask_with_no_search_results(self, chat_engine, mock_search_engine, mock_router):
        mock_search_engine.search.return_value = []
        mock_router.complete.return_value = "검색 결과가 없어 답변이 어렵습니다."
        session = ChatSession()
        response = chat_engine.ask("알 수 없는 질문", session=session)
        assert response.content is not None

    def test_ask_extracts_sources(self, chat_engine):
        session = ChatSession()
        response = chat_engine.ask("Python async", session=session)
        assert len(response.sources) >= 1


class TestContextAssembly:
    def test_context_includes_search_results(self, chat_engine, mock_router):
        session = ChatSession()
        chat_engine.ask("Python", session=session)
        prompt = mock_router.complete.call_args[0][0]
        assert "Python" in prompt
        assert "20260101-test-abc1" in prompt

    def test_context_no_question_line(self, chat_engine, mock_router):
        """context에 'Question:' 줄이 없어야 함."""
        session = ChatSession()
        chat_engine.ask("Python", session=session)
        prompt = mock_router.complete.call_args[0][0]
        assert "Question:" not in prompt
        assert "Summary:" in prompt

    def test_sources_use_summary_not_question(self, chat_engine):
        """sources에 question 대신 summary 키가 있어야 함."""
        session = ChatSession()
        response = chat_engine.ask("Python async", session=session)
        for src in response.sources:
            assert "question" not in src
            assert "summary" in src

    def test_citation_format_no_question(self, chat_engine, mock_router):
        """citation 안내에 'question' 참조가 없어야 함."""
        session = ChatSession()
        chat_engine.ask("Python", session=session)
        prompt = mock_router.complete.call_args[0][0]
        assert "— question" not in prompt
