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

# Encoding fallback chain (strict): UTF-8 (with BOM) → CP949 (Korean legacy)
_FALLBACK_ENCODINGS = ("utf-8-sig", "cp949")


def read_text_with_fallback(path: Path) -> str:
    """Read a text file, trying multiple encodings.

    Tries UTF-8 (with BOM support) first, then CP949 (Korean legacy superset of EUC-KR),
    then UTF-8 with errors='replace' for mostly-UTF-8 files with a few corrupted bytes.
    """
    raw = path.read_bytes()
    for encoding in _FALLBACK_ENCODINGS:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    # Last resort: UTF-8 with replacement characters for corrupted bytes.
    # Preferable to latin-1 for Korean content (preserves valid Korean characters).
    return raw.decode("utf-8", errors="replace")
