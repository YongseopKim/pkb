"""Markdown generator for PKB conversations."""

from pathlib import Path

import yaml

from pkb.models.jsonl import Conversation


def conversation_to_markdown(conv: Conversation, bundle_id: str) -> str:
    """Convert a Conversation to formatted Markdown.

    Generates a Markdown document with a header and all turns formatted
    as ## User / ## Assistant sections.
    """
    lines = [
        f"# {conv.meta.title or bundle_id}",
        "",
        f"- **Bundle**: `{bundle_id}`",
        f"- **Platform**: {conv.meta.platform}",
        f"- **Exported**: {conv.meta.exported_at.isoformat()}",
        "",
        "---",
        "",
    ]

    for turn in conv.turns:
        role_label = turn.role.capitalize()
        lines.append(f"## {role_label}")
        lines.append("")
        lines.append(turn.content)
        lines.append("")

    return "\n".join(lines)


def write_md_file(
    conv: Conversation,
    bundle_id: str,
    frontmatter: dict,
    output_path: Path,
) -> None:
    """Write a Markdown file with YAML frontmatter.

    Args:
        conv: Parsed conversation.
        bundle_id: Bundle identifier.
        frontmatter: Dict to serialize as YAML frontmatter.
        output_path: Where to write the file.
    """
    md_body = conversation_to_markdown(conv, bundle_id)
    fm_text = yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False)
    content = f"---\n{fm_text}---\n\n{md_body}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
