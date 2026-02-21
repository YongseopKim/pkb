"""Tests for chat models."""

from pkb.chat.models import ChatMessage, ChatResponse, ChatSession


class TestChatMessage:
    def test_user_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message(self):
        msg = ChatMessage(role="assistant", content="Hi there")
        assert msg.role == "assistant"


class TestChatResponse:
    def test_response_with_sources(self):
        resp = ChatResponse(
            content="The answer is 42.",
            sources=[
                {"bundle_id": "20260101-test-abc1", "question": "What is 42?"},
            ],
        )
        assert len(resp.sources) == 1
        assert resp.sources[0]["bundle_id"] == "20260101-test-abc1"

    def test_response_without_sources(self):
        resp = ChatResponse(content="I don't know.", sources=[])
        assert len(resp.sources) == 0


class TestChatSession:
    def test_add_message(self):
        session = ChatSession()
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi")
        assert len(session.messages) == 2

    def test_get_history(self):
        session = ChatSession()
        for i in range(10):
            session.add_message("user", f"msg {i}")
        history = session.get_history(max_messages=4)
        assert len(history) == 4
        assert history[-1].content == "msg 9"

    def test_empty_session(self):
        session = ChatSession()
        assert len(session.messages) == 0
        assert session.get_history() == []
