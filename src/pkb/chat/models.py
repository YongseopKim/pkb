"""Chat data models."""

from __future__ import annotations

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str  # "user" or "assistant"
    content: str


class ChatResponse(BaseModel):
    """Response from the chat engine."""

    content: str
    sources: list[dict] = []


class ChatSession:
    """Maintains conversation history with sliding window."""

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(ChatMessage(role=role, content=content))

    def get_history(self, max_messages: int = 10) -> list[ChatMessage]:
        """Get recent messages (sliding window)."""
        if not self.messages:
            return []
        return self.messages[-max_messages:]
