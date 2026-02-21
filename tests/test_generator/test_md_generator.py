"""Tests for Markdown generator."""

from datetime import datetime, timezone

from pkb.generator.md_generator import conversation_to_markdown, write_md_file
from pkb.models.jsonl import Conversation, ConversationMeta, Turn


def _make_conversation(
    platform: str = "claude",
    title: str = "테스트 대화",
    turns: list[Turn] | None = None,
) -> Conversation:
    if turns is None:
        turns = [
            Turn(role="user", content="파이썬에서 async가 뭐야?"),
            Turn(role="assistant", content="async는 비동기 프로그래밍 키워드입니다."),
        ]
    return Conversation(
        meta=ConversationMeta(
            platform=platform,
            url="https://example.com/chat/123",
            exported_at=datetime(2026, 2, 21, 6, 0, 0, tzinfo=timezone.utc),
            title=title,
        ),
        turns=turns,
    )


class TestConversationToMarkdown:
    def test_basic_output(self):
        conv = _make_conversation()
        md = conversation_to_markdown(conv, "20260221-test-slug-a3f2")
        assert "## User" in md
        assert "## Assistant" in md
        assert "파이썬에서 async가 뭐야?" in md
        assert "async는 비동기 프로그래밍 키워드입니다." in md

    def test_includes_bundle_id_header(self):
        conv = _make_conversation()
        md = conversation_to_markdown(conv, "20260221-test-slug-a3f2")
        assert "20260221-test-slug-a3f2" in md

    def test_multi_turn(self):
        turns = [
            Turn(role="user", content="질문 1"),
            Turn(role="assistant", content="답변 1"),
            Turn(role="user", content="질문 2"),
            Turn(role="assistant", content="답변 2"),
        ]
        conv = _make_conversation(turns=turns)
        md = conversation_to_markdown(conv, "test-bundle")
        assert md.count("## User") == 2
        assert md.count("## Assistant") == 2

    def test_consecutive_assistant_turns(self):
        """Claude/Perplexity can have consecutive assistant turns."""
        turns = [
            Turn(role="user", content="질문"),
            Turn(role="assistant", content="생각 중..."),
            Turn(role="assistant", content="실제 답변"),
        ]
        conv = _make_conversation(turns=turns)
        md = conversation_to_markdown(conv, "test-bundle")
        assert "생각 중..." in md
        assert "실제 답변" in md


class TestWriteMdFile:
    def test_creates_file(self, tmp_path):
        conv = _make_conversation()
        frontmatter = {
            "model": "claude-sonnet-4",
            "summary": "async 설명",
            "platform": "claude",
        }
        output = tmp_path / "claude.md"
        write_md_file(conv, "test-bundle", frontmatter, output)
        assert output.exists()
        content = output.read_text()
        assert content.startswith("---\n")
        assert "model: claude-sonnet-4" in content
        assert "summary:" in content

    def test_frontmatter_and_content(self, tmp_path):
        conv = _make_conversation()
        frontmatter = {"platform": "claude"}
        output = tmp_path / "claude.md"
        write_md_file(conv, "test-bundle", frontmatter, output)
        content = output.read_text()
        # Should have frontmatter delimiters
        parts = content.split("---\n")
        assert len(parts) >= 3  # '', frontmatter, content
