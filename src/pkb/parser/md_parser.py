"""Markdown parser for llm-chat-exporter MD output.

Parses MD exports with graceful degradation:
  Level 1: # [Platform](URL) header + ## LLM 응답 N sections
  Level 2: Any ## headings → split into sections as assistant turns
  Level 3: No structure → entire content as single assistant turn
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from pkb.constants import PLATFORMS
from pkb.models.jsonl import Conversation, ConversationMeta, Turn
from pkb.parser.exceptions import MDParseError

# Regex for header: # [DisplayName](URL)
_HEADER_RE = re.compile(r"^#\s+\[([^\]]+)\]\(([^)]+)\)\s*$")

# Regex for Level 1 section: ## LLM 응답 N
_LEVEL1_RE = re.compile(r"^##\s+LLM\s+응답\s+\d+", re.MULTILINE)

# Regex for any Level 2 heading: ## Anything
_LEVEL2_RE = re.compile(r"^##\s+", re.MULTILINE)

# Double separator pattern
_DOUBLE_SEP_RE = re.compile(r"(?:^|\n)---\s*\n---\s*(?:\n|$)")

# Known URL domain → platform mapping
_DOMAIN_TO_PLATFORM: dict[str, str] = {
    "claude.ai": "claude",
    "chatgpt.com": "chatgpt",
    "chat.openai.com": "chatgpt",
    "gemini.google.com": "gemini",
    "grok.com": "grok",
    "perplexity.ai": "perplexity",
    "www.perplexity.ai": "perplexity",
}


def parse_md_string(
    data: str,
    *,
    platform: str | None = None,
    exported_at: datetime | None = None,
) -> Conversation:
    """Parse a markdown string into a Conversation.

    Args:
        data: Raw markdown content.
        platform: Override platform name (header link takes priority if present).
        exported_at: Override exported_at timestamp.

    Returns:
        Conversation with metadata and turns.

    Raises:
        MDParseError: If data is empty or whitespace-only.
    """
    stripped = data.strip()
    if not stripped:
        raise MDParseError("Empty markdown data")

    lines = stripped.split("\n")
    first_line = lines[0]

    # Try extracting header
    header_platform, header_url = _extract_header(first_line)

    # Determine platform: header > param > "unknown"
    resolved_platform = header_platform or platform or "unknown"

    # Determine URL
    resolved_url = header_url

    # Remove header line from content if it was a valid header
    if header_platform is not None or header_url is not None:
        content = "\n".join(lines[1:])
    else:
        content = stripped

    # Strip double separators
    content = _DOUBLE_SEP_RE.sub("\n", content).strip()

    # Try Level 1 splitting
    sections = _split_sections_level1(content)

    # Fall back to Level 2
    if sections is None:
        sections = _split_sections_level2(content)

    # Fall back to Level 3 (whole content as single section)
    if sections is None:
        sections = [content]

    # Convert sections to turns
    turns = _sections_to_turns(sections)

    # Ensure at least one turn (edge case: all sections were empty after stripping)
    if not turns and content.strip():
        turns = [Turn(role="assistant", content=content.strip(), timestamp=None)]

    # Build metadata
    meta = ConversationMeta(
        platform=resolved_platform,
        url=resolved_url,
        exported_at=exported_at or datetime.now(tz=timezone.utc),
        title=None,
    )

    return Conversation(meta=meta, turns=turns)


def parse_md_file(
    path: Path,
    *,
    platform: str | None = None,
) -> Conversation:
    """Parse a markdown file into a Conversation.

    Platform detection priority: header link > platform param > filename stem.
    exported_at: file mtime.

    Args:
        path: Path to the .md file.
        platform: Override platform name.

    Returns:
        Conversation with metadata and turns.

    Raises:
        FileNotFoundError: If file doesn't exist.
        MDParseError: If file content is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = path.read_text(encoding="utf-8-sig")

    # If no explicit platform param, try filename
    if platform is None:
        filename_platform = _platform_from_filename(path)
        # Only use filename platform if it's a known platform
        if filename_platform in PLATFORMS:
            platform = filename_platform

    # Use file mtime as exported_at
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    return parse_md_string(data, platform=platform, exported_at=mtime)


# ── Internal helpers ─────────────────────────────────────────────────


def _extract_header(first_line: str) -> tuple[str | None, str | None]:
    """Extract platform and URL from a header line like # [Platform](URL).

    Returns:
        (platform, url) tuple. Both None if not a valid header.
    """
    match = _HEADER_RE.match(first_line.strip())
    if not match:
        return None, None

    url = match.group(2)
    platform = _platform_from_url(url)
    return platform, url


def _platform_from_url(url: str | None) -> str | None:
    """Determine platform from URL domain."""
    if url is None:
        return None

    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return None

    return _DOMAIN_TO_PLATFORM.get(host)


def _platform_from_filename(path: Path) -> str:
    """Extract platform name from filename stem (e.g., claude.md → claude)."""
    return path.stem.lower()


def _split_sections_level1(content: str) -> list[str] | None:
    """Split content by ## LLM 응답 N pattern (Level 1).

    Returns:
        List of section contents, or None if pattern not found.
    """
    matches = list(_LEVEL1_RE.finditer(content))
    if not matches:
        return None

    sections = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section = content[start:end]
        # Strip double separators and surrounding whitespace
        section = _DOUBLE_SEP_RE.sub("\n", section).strip()
        sections.append(section)

    return sections


def _split_sections_level2(content: str) -> list[str] | None:
    """Split content by any ## heading (Level 2).

    Returns:
        List of section contents, or None if no headings found.
    """
    matches = list(_LEVEL2_RE.finditer(content))
    if not matches:
        return None

    sections = []
    for i, match in enumerate(matches):
        # Find end of this heading line
        heading_end = content.find("\n", match.start())
        if heading_end == -1:
            heading_end = len(content)
        start = heading_end + 1

        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section = content[start:end].strip()
        sections.append(section)

    return sections


def _sections_to_turns(sections: list[str]) -> list[Turn]:
    """Convert content sections to Turn objects (all role=assistant)."""
    turns = []
    for section in sections:
        stripped = section.strip()
        if stripped:
            turns.append(Turn(role="assistant", content=stripped, timestamp=None))
    return turns
