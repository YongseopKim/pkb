"""Tests for JSONL data models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pkb.models.jsonl import Conversation, ConversationMeta, Turn


class TestConversationMeta:
    def test_valid_meta(self):
        meta = ConversationMeta(
            platform="claude",
            url="https://claude.ai/chat/abc",
            exported_at=datetime(2026, 2, 21, 6, 2, 42, tzinfo=timezone.utc),
            title="Test conversation",
        )
        assert meta.platform == "claude"
        assert meta.title == "Test conversation"

    def test_missing_platform_raises(self):
        with pytest.raises(ValidationError):
            ConversationMeta(
                url="https://example.com",
                exported_at=datetime.now(tz=timezone.utc),
                title="test",
            )

    def test_optional_title(self):
        meta = ConversationMeta(
            platform="chatgpt",
            url="https://chatgpt.com/c/abc",
            exported_at=datetime(2026, 2, 21, tzinfo=timezone.utc),
            title=None,
        )
        assert meta.title is None

    def test_optional_url_defaults_none(self):
        meta = ConversationMeta(
            platform="claude",
            exported_at=datetime(2026, 2, 21, tzinfo=timezone.utc),
        )
        assert meta.url is None

    def test_url_with_value_works(self):
        meta = ConversationMeta(
            platform="claude",
            url="https://claude.ai/chat/abc",
            exported_at=datetime(2026, 2, 21, tzinfo=timezone.utc),
        )
        assert meta.url == "https://claude.ai/chat/abc"


class TestTurn:
    def test_valid_turn(self):
        turn = Turn(
            role="user",
            content="Hello",
            timestamp=datetime(2026, 2, 21, 6, 0, 0, tzinfo=timezone.utc),
        )
        assert turn.role == "user"
        assert turn.content == "Hello"

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            Turn(
                role="system",
                content="Hello",
                timestamp=datetime.now(tz=timezone.utc),
            )

    def test_optional_timestamp(self):
        turn = Turn(role="assistant", content="Hi", timestamp=None)
        assert turn.timestamp is None


class TestConversation:
    def _make_conversation(self, turns=None):
        meta = ConversationMeta(
            platform="claude",
            url="https://claude.ai/chat/abc",
            exported_at=datetime(2026, 2, 21, 6, 0, 0, tzinfo=timezone.utc),
            title="Test",
        )
        if turns is None:
            turns = [
                Turn(role="user", content="Q1", timestamp=None),
                Turn(role="assistant", content="A1", timestamp=None),
            ]
        return Conversation(meta=meta, turns=turns)

    def test_basic_conversation(self):
        conv = self._make_conversation()
        assert len(conv.turns) == 2
        assert conv.meta.platform == "claude"

    def test_first_user_message(self):
        conv = self._make_conversation()
        assert conv.first_user_message == "Q1"

    def test_first_user_message_empty_turns(self):
        conv = self._make_conversation(turns=[])
        assert conv.first_user_message is None

    def test_first_user_message_no_user_turn(self):
        turns = [Turn(role="assistant", content="A1", timestamp=None)]
        conv = self._make_conversation(turns=turns)
        assert conv.first_user_message is None

    def test_turn_count(self):
        conv = self._make_conversation()
        assert conv.turn_count == 2

    def test_consecutive_assistant_turns_allowed(self):
        """Claude/Perplexity have consecutive assistant turns."""
        turns = [
            Turn(role="user", content="Q", timestamp=None),
            Turn(role="assistant", content="thinking...", timestamp=None),
            Turn(role="assistant", content="actual answer", timestamp=None),
        ]
        conv = self._make_conversation(turns=turns)
        assert conv.turn_count == 3
