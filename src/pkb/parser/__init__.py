"""Parsing modules for JSONL and MD formats."""

import logging
from pathlib import Path

from pkb.parser.jsonl_parser import parse_jsonl_file, parse_jsonl_string
from pkb.parser.md_parser import parse_md_file, parse_md_string

logger = logging.getLogger(__name__)

__all__ = [
    "parse_jsonl_file",
    "parse_jsonl_string",
    "parse_md_file",
    "parse_md_string",
    "read_text_with_fallback",
]

# Encoding fallback chain: UTF-8 (with BOM) → CP949 (Korean legacy) → Latin-1 (always succeeds)
_FALLBACK_ENCODINGS = ("utf-8-sig", "cp949", "latin-1")


def read_text_with_fallback(path: Path) -> str:
    """Read a text file, trying multiple encodings.

    Tries UTF-8 (with BOM support) first, then CP949 (Korean legacy superset of EUC-KR),
    then Latin-1 (always succeeds as a last resort).
    """
    raw = path.read_bytes()
    for encoding in _FALLBACK_ENCODINGS:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    # latin-1 never fails, so this is unreachable, but just in case:
    return raw.decode("latin-1")  # pragma: no cover
