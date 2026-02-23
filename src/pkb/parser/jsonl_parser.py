"""JSONL parser for llm-chat-exporter output."""

import json
from pathlib import Path

from pkb.models.jsonl import Conversation, ConversationMeta, Turn
from pkb.parser.exceptions import MetaLineError, ParseError, TurnParseError


def parse_jsonl_string(data: str) -> Conversation:
    """Parse a JSONL string into a Conversation.

    The first line must be a _meta object. Subsequent lines are turns.
    Consecutive same-role turns are allowed (Claude thinking, Perplexity multi-answer).
    """
    lines = [line for line in data.strip().split("\n") if line.strip()]

    if not lines:
        raise ParseError("Empty JSONL data")

    # Parse meta line (line 1)
    meta = _parse_meta_line(lines[0])

    # Parse turn lines (line 2+)
    turns: list[Turn] = []
    for i, line in enumerate(lines[1:], start=2):
        turn = _parse_turn_line(line, line_number=i)
        turns.append(turn)

    return Conversation(meta=meta, turns=turns)


def parse_jsonl_file(path: Path) -> Conversation:
    """Parse a JSONL file into a Conversation.

    Uses utf-8-sig encoding to handle BOM.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    from pkb.parser import read_text_with_fallback

    data = read_text_with_fallback(path)
    return parse_jsonl_string(data)


def _parse_meta_line(line: str) -> ConversationMeta:
    """Parse the first line as a _meta object."""
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as e:
        raise MetaLineError(f"Invalid JSON in meta line: {e}") from e

    if not raw.get("_meta"):
        raise MetaLineError("First line must have '_meta': true")

    try:
        return ConversationMeta(
            platform=raw["platform"],
            url=raw["url"],
            exported_at=raw["exported_at"],
            title=raw.get("title"),
        )
    except (KeyError, ValueError) as e:
        raise MetaLineError(f"Invalid meta fields: {e}") from e


def _parse_turn_line(line: str, *, line_number: int) -> Turn:
    """Parse a single turn line."""
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as e:
        raise TurnParseError(f"Invalid JSON: {e}", line_number=line_number) from e

    if "role" not in raw:
        raise TurnParseError("Missing 'role' field", line_number=line_number)

    try:
        return Turn(
            role=raw["role"],
            content=raw.get("content", ""),
            timestamp=raw.get("timestamp"),
        )
    except ValueError as e:
        raise TurnParseError(str(e), line_number=line_number) from e
