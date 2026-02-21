"""Parse YAML frontmatter from Markdown files."""

from pathlib import Path

import yaml


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split text into frontmatter YAML and body.

    Returns:
        Tuple of (frontmatter_yaml, body).

    Raises:
        ValueError: If no valid frontmatter block is found.
    """
    if not text.startswith("---"):
        raise ValueError("No frontmatter found: file does not start with '---'")

    # Find closing delimiter (second ---)
    end_idx = text.find("\n---", 3)
    if end_idx == -1:
        raise ValueError("No closing '---' delimiter for frontmatter")

    fm_yaml = text[4:end_idx]  # skip opening "---\n"
    body = text[end_idx + 4:]  # skip "\n---"
    return fm_yaml, body


def parse_frontmatter(md_path: Path) -> dict:
    """Parse YAML frontmatter from a Markdown file.

    Args:
        md_path: Path to the Markdown file.

    Returns:
        Dictionary of frontmatter fields.

    Raises:
        ValueError: If frontmatter is missing or malformed.
    """
    text = md_path.read_text(encoding="utf-8")
    fm_yaml, _ = _split_frontmatter(text)
    result = yaml.safe_load(fm_yaml)
    return result if result is not None else {}


def parse_md_body(md_path: Path) -> str:
    """Extract the body (post-frontmatter) from a Markdown file.

    Args:
        md_path: Path to the Markdown file.

    Returns:
        Body text after the frontmatter block.

    Raises:
        ValueError: If frontmatter is missing or malformed.
    """
    text = md_path.read_text(encoding="utf-8")
    _, body = _split_frontmatter(text)
    return body
