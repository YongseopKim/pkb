"""Parser exceptions."""


class ParseError(Exception):
    """Base exception for parsing errors."""


class MetaLineError(ParseError):
    """Error in the _meta line (first line) of a JSONL file."""


class TurnParseError(ParseError):
    """Error parsing a conversation turn line."""

    def __init__(self, message: str, line_number: int):
        self.line_number = line_number
        super().__init__(f"Line {line_number}: {message}")


class MDParseError(ParseError):
    """Error parsing a Markdown export file."""
