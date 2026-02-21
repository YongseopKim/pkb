"""Prompt template loading and rendering for PKB."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt template by name (without .txt extension).

    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, **kwargs: str) -> str:
    """Render a prompt template with the given variables.

    Uses str.format_map with a defaultdict-like approach to leave
    unmatched placeholders as-is (useful for JSON template braces).
    """

    class SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(SafeDict(**kwargs))
