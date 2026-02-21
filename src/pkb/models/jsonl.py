"""Data models for JSONL conversation format."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ConversationMeta(BaseModel):
    """Metadata from the first line of a JSONL file."""

    platform: str
    url: str | None = None
    exported_at: datetime
    title: str | None = None


class Turn(BaseModel):
    """A single conversation turn (user or assistant)."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime | None = None


class Conversation(BaseModel):
    """A parsed JSONL conversation."""

    meta: ConversationMeta
    turns: list[Turn]

    @property
    def first_user_message(self) -> str | None:
        """First user message content, used for bundle ID generation."""
        for turn in self.turns:
            if turn.role == "user":
                return turn.content
        return None

    @property
    def turn_count(self) -> int:
        return len(self.turns)
