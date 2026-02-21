"""Tests for frontmatter parser."""

import pytest

from pkb.generator.frontmatter_parser import parse_frontmatter, parse_md_body


class TestParseFrontmatter:
    """Tests for parse_frontmatter()."""

    def test_valid_frontmatter(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text(
            "---\n"
            "id: 20260101-test-abc1\n"
            "question: 테스트 질문\n"
            "summary: 테스트 요약\n"
            "domains:\n"
            "  - dev\n"
            "  - ai\n"
            "topics:\n"
            "  - python\n"
            "---\n"
            "\n# Content\nHello\n",
            encoding="utf-8",
        )
        result = parse_frontmatter(md)
        assert result["id"] == "20260101-test-abc1"
        assert result["question"] == "테스트 질문"
        assert result["summary"] == "테스트 요약"
        assert result["domains"] == ["dev", "ai"]
        assert result["topics"] == ["python"]

    def test_frontmatter_with_all_fields(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text(
            "---\n"
            "id: 20260101-full-abc1\n"
            "question: 전체 필드 테스트\n"
            "summary: 전체 요약\n"
            "slug: full-test\n"
            "domains:\n"
            "  - dev\n"
            "topics:\n"
            "  - python\n"
            "pending_topics:\n"
            "  - new-topic\n"
            "platforms:\n"
            "  - claude\n"
            "created_at: '2026-01-01T00:00:00+00:00'\n"
            "consensus: 동의\n"
            "divergence: null\n"
            "---\n",
            encoding="utf-8",
        )
        result = parse_frontmatter(md)
        assert result["slug"] == "full-test"
        assert result["pending_topics"] == ["new-topic"]
        assert result["platforms"] == ["claude"]
        assert result["consensus"] == "동의"
        assert result["divergence"] is None

    def test_missing_closing_delimiter(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text(
            "---\n"
            "id: test\n"
            "question: no closing\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="closing"):
            parse_frontmatter(md)

    def test_no_frontmatter(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Just a heading\nNo frontmatter here.\n", encoding="utf-8")
        with pytest.raises(ValueError, match="frontmatter"):
            parse_frontmatter(md)

    def test_empty_frontmatter(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("---\n---\n\nContent\n", encoding="utf-8")
        result = parse_frontmatter(md)
        assert result == {} or result is None

    def test_unicode_content(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text(
            "---\n"
            "id: 20260101-한글-abc1\n"
            "question: 한글 질문입니다\n"
            "summary: 유니코드 요약 テスト\n"
            "---\n",
            encoding="utf-8",
        )
        result = parse_frontmatter(md)
        assert result["question"] == "한글 질문입니다"
        assert "テスト" in result["summary"]


class TestParseMdBody:
    """Tests for parse_md_body()."""

    def test_body_after_frontmatter(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text(
            "---\nid: test\n---\n\n# Heading\n\nBody content here.\n",
            encoding="utf-8",
        )
        body = parse_md_body(md)
        assert "# Heading" in body
        assert "Body content here." in body
        assert "---" not in body

    def test_body_preserves_whitespace(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text(
            "---\nid: test\n---\n\nLine 1\n\nLine 2\n",
            encoding="utf-8",
        )
        body = parse_md_body(md)
        assert "Line 1\n\nLine 2" in body

    def test_no_body(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("---\nid: test\n---\n", encoding="utf-8")
        body = parse_md_body(md)
        assert body.strip() == ""

    def test_no_frontmatter_returns_whole_content(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# No frontmatter\nAll content.\n", encoding="utf-8")
        with pytest.raises(ValueError, match="frontmatter"):
            parse_md_body(md)
