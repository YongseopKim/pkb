"""Tests for MD parser."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from pkb.parser.exceptions import MDParseError
from pkb.parser.md_parser import (
    _extract_header,
    _platform_from_filename,
    _platform_from_url,
    _sections_to_turns,
    _split_sections_level1,
    _split_sections_level2,
    parse_md_file,
    parse_md_string,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MD_SAMPLES_DIR = REPO_ROOT / "exporter-examples" / "md" / "PKB"


# ── Header extraction ────────────────────────────────────────────────


class TestExtractHeader:
    def test_claude_header(self):
        platform, url = _extract_header(
            "# [Claude](https://claude.ai/chat/0cc02326-019d-4600-8435-71987826ccf9)"
        )
        assert platform == "claude"
        assert "claude.ai" in url

    def test_chatgpt_header(self):
        platform, url = _extract_header(
            "# [ChatGPT](https://chatgpt.com/g/g-p-699294519bec8191be66cfbabb8dc419-saideu-peurojegteu-pkb/c/69992dc0-5fe0-83a3-bfd2-7c824aac862c)"
        )
        assert platform == "chatgpt"
        assert "chatgpt.com" in url

    def test_gemini_header(self):
        platform, url = _extract_header(
            "# [Gemini](https://gemini.google.com/app/557a45b7b7de5871?hl=ko)"
        )
        assert platform == "gemini"
        assert "gemini.google.com" in url

    def test_grok_header(self):
        platform, url = _extract_header(
            "# [Grok](https://grok.com/c/59ead090-d0d4-4085-b370-610a81224354)"
        )
        assert platform == "grok"
        assert "grok.com" in url

    def test_perplexity_header(self):
        platform, url = _extract_header(
            "# [Perplexity](https://www.perplexity.ai/search/abc)"
        )
        assert platform == "perplexity"
        assert "perplexity.ai" in url

    def test_no_link_header(self):
        """Plain heading without link."""
        platform, url = _extract_header("# Some Title")
        assert platform is None
        assert url is None

    def test_empty_line(self):
        platform, url = _extract_header("")
        assert platform is None
        assert url is None

    def test_not_a_heading(self):
        """Line that doesn't start with # ."""
        platform, url = _extract_header("## Subheading")
        assert platform is None
        assert url is None

    def test_unknown_platform_url(self):
        """Unknown domain in header link — still extracts URL but platform comes from URL."""
        platform, url = _extract_header("# [MyLLM](https://unknown.example.com/chat)")
        assert url == "https://unknown.example.com/chat"
        assert platform is None  # unknown platform


# ── Platform from URL ────────────────────────────────────────────────


class TestPlatformFromUrl:
    def test_claude_url(self):
        assert _platform_from_url("https://claude.ai/chat/abc") == "claude"

    def test_chatgpt_url(self):
        assert _platform_from_url("https://chatgpt.com/c/abc") == "chatgpt"

    def test_gemini_url(self):
        assert _platform_from_url("https://gemini.google.com/app/abc") == "gemini"

    def test_grok_url(self):
        assert _platform_from_url("https://grok.com/c/abc") == "grok"

    def test_perplexity_url(self):
        assert _platform_from_url("https://www.perplexity.ai/search/abc") == "perplexity"

    def test_unknown_url(self):
        assert _platform_from_url("https://unknown.example.com/chat") is None

    def test_none_url(self):
        assert _platform_from_url(None) is None


# ── Platform from filename ───────────────────────────────────────────


class TestPlatformFromFilename:
    def test_claude_filename(self):
        assert _platform_from_filename(Path("claude.md")) == "claude"

    def test_chatgpt_filename(self):
        assert _platform_from_filename(Path("/path/to/chatgpt.md")) == "chatgpt"

    def test_unknown_filename(self):
        assert _platform_from_filename(Path("notes.md")) == "notes"


# ── Level 1 splitting (## LLM 응답 N) ───────────────────────────────


class TestSplitSectionsLevel1:
    def test_single_response(self):
        content = "## LLM 응답 1\n\nHello world\n"
        sections = _split_sections_level1(content)
        assert len(sections) == 1
        assert "Hello world" in sections[0]

    def test_multiple_responses(self):
        content = (
            "## LLM 응답 1\n\nFirst response\n\n"
            "---\n---\n\n"
            "## LLM 응답 2\n\nSecond response\n"
        )
        sections = _split_sections_level1(content)
        assert len(sections) == 2
        assert "First response" in sections[0]
        assert "Second response" in sections[1]

    def test_preserves_markdown_in_content(self):
        content = "## LLM 응답 1\n\n### Subheading\n\n- bullet\n- list\n\n```python\ncode\n```\n"
        sections = _split_sections_level1(content)
        assert len(sections) == 1
        assert "### Subheading" in sections[0]
        assert "```python" in sections[0]

    def test_no_llm_response_pattern(self):
        content = "## Some other heading\n\nContent here\n"
        sections = _split_sections_level1(content)
        assert sections is None  # signal to fall back to Level 2


class TestSplitSectionsLevel2:
    def test_generic_headings(self):
        content = "## First Section\n\nContent 1\n\n## Second Section\n\nContent 2\n"
        sections = _split_sections_level2(content)
        assert len(sections) == 2

    def test_no_headings(self):
        content = "Just plain text\nwith multiple lines\n"
        sections = _split_sections_level2(content)
        assert sections is None  # signal to fall back to Level 3


# ── Sections to turns ────────────────────────────────────────────────


class TestSectionsToTurns:
    def test_basic_sections(self):
        sections = ["Content 1", "Content 2"]
        turns = _sections_to_turns(sections)
        assert len(turns) == 2
        assert all(t.role == "assistant" for t in turns)
        assert turns[0].content == "Content 1"

    def test_empty_sections_filtered(self):
        sections = ["Content", "", "  ", "More content"]
        turns = _sections_to_turns(sections)
        assert len(turns) == 2

    def test_whitespace_stripped(self):
        sections = ["  Content with spaces  "]
        turns = _sections_to_turns(sections)
        assert turns[0].content == "Content with spaces"


# ── parse_md_string ──────────────────────────────────────────────────


class TestParseMdString:
    def test_level1_full_structure(self):
        """Full Level 1: header + ## LLM 응답 N sections."""
        md = (
            "# [Claude](https://claude.ai/chat/abc)\n\n"
            "---\n---\n\n"
            "## LLM 응답 1\n\n"
            "First response content\n\n"
            "---\n---\n\n"
            "## LLM 응답 2\n\n"
            "Second response content\n"
        )
        conv = parse_md_string(md)
        assert conv.meta.platform == "claude"
        assert conv.meta.url == "https://claude.ai/chat/abc"
        assert len(conv.turns) == 2
        assert "First response" in conv.turns[0].content
        assert "Second response" in conv.turns[1].content

    def test_level2_generic_headings(self):
        """Level 2: header + generic ## headings (no LLM 응답 pattern)."""
        md = (
            "# [ChatGPT](https://chatgpt.com/c/abc)\n\n"
            "## Analysis\n\nSome analysis\n\n"
            "## Conclusion\n\nSome conclusion\n"
        )
        conv = parse_md_string(md)
        assert conv.meta.platform == "chatgpt"
        assert len(conv.turns) == 2

    def test_level3_plain_text(self):
        """Level 3: no structure at all."""
        md = "Just some plain text\nwith multiple lines\nand no structure.\n"
        conv = parse_md_string(md)
        assert conv.meta.platform == "unknown"
        assert len(conv.turns) == 1
        assert "plain text" in conv.turns[0].content

    def test_level3_header_no_sections(self):
        """Level 3: header present but no ## sections."""
        md = (
            "# [Gemini](https://gemini.google.com/app/abc)\n\n"
            "Just plain content without any sections.\n"
            "More content here.\n"
        )
        conv = parse_md_string(md)
        assert conv.meta.platform == "gemini"
        assert len(conv.turns) == 1

    def test_platform_param_overrides_filename(self):
        md = "Some content\n"
        conv = parse_md_string(md, platform="grok")
        assert conv.meta.platform == "grok"

    def test_header_overrides_platform_param(self):
        md = "# [Claude](https://claude.ai/chat/abc)\n\nContent\n"
        conv = parse_md_string(md, platform="chatgpt")
        assert conv.meta.platform == "claude"

    def test_exported_at_from_param(self):
        dt = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        conv = parse_md_string("Content\n", exported_at=dt)
        assert conv.meta.exported_at == dt

    def test_exported_at_defaults_to_now(self):
        before = datetime.now(tz=timezone.utc)
        conv = parse_md_string("Content\n")
        after = datetime.now(tz=timezone.utc)
        assert before <= conv.meta.exported_at <= after

    def test_url_none_when_no_header(self):
        conv = parse_md_string("Just text\n")
        assert conv.meta.url is None

    def test_empty_string_raises(self):
        with pytest.raises(MDParseError):
            parse_md_string("")

    def test_whitespace_only_raises(self):
        with pytest.raises(MDParseError):
            parse_md_string("   \n  \n  ")

    def test_double_separator_stripped(self):
        """Double --- separator between header and content should not appear in turns."""
        md = (
            "# [Claude](https://claude.ai/chat/abc)\n\n"
            "---\n---\n\n"
            "## LLM 응답 1\n\nContent here\n"
        )
        conv = parse_md_string(md)
        assert len(conv.turns) == 1
        assert "---" not in conv.turns[0].content


# ── parse_md_file ────────────────────────────────────────────────────


class TestParseMdFile:
    def test_parse_file(self, tmp_path: Path):
        md_file = tmp_path / "claude.md"
        md_file.write_text(
            "# [Claude](https://claude.ai/chat/abc)\n\n"
            "---\n---\n\n"
            "## LLM 응답 1\n\nContent\n",
            encoding="utf-8",
        )
        conv = parse_md_file(md_file)
        assert conv.meta.platform == "claude"
        assert len(conv.turns) == 1

    def test_platform_from_filename(self, tmp_path: Path):
        """No header → platform inferred from filename."""
        md_file = tmp_path / "gemini.md"
        md_file.write_text("Just content\n", encoding="utf-8")
        conv = parse_md_file(md_file)
        assert conv.meta.platform == "gemini"

    def test_platform_param_overrides_filename(self, tmp_path: Path):
        md_file = tmp_path / "notes.md"
        md_file.write_text("Content\n", encoding="utf-8")
        conv = parse_md_file(md_file, platform="claude")
        assert conv.meta.platform == "claude"

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_md_file(tmp_path / "nonexistent.md")

    def test_exported_at_from_file_mtime(self, tmp_path: Path):
        md_file = tmp_path / "test.md"
        md_file.write_text("Content\n", encoding="utf-8")
        conv = parse_md_file(md_file)
        # Should use file mtime, not "now"
        assert conv.meta.exported_at is not None

    def test_euc_kr_encoded_file(self, tmp_path: Path):
        """EUC-KR (Korean legacy) encoded file should parse without error."""
        korean_text = "## 트레바리 넥스랩\n\n한글 내용입니다.\n"
        md_file = tmp_path / "evernote_note.md"
        md_file.write_bytes(korean_text.encode("euc-kr"))
        conv = parse_md_file(md_file)
        assert len(conv.turns) >= 1
        assert "한글" in conv.turns[0].content

    def test_cp949_encoded_file(self, tmp_path: Path):
        """CP949 (Windows Korean) encoded file should parse without error."""
        korean_text = "# 박진영 노트\n\n내용입니다.\n"
        md_file = tmp_path / "note.md"
        md_file.write_bytes(korean_text.encode("cp949"))
        conv = parse_md_file(md_file)
        assert len(conv.turns) >= 1

    def test_latin1_encoded_file(self, tmp_path: Path):
        """Latin-1 encoded file should parse without error."""
        text = "## Café résumé\n\nContent with accents.\n"
        md_file = tmp_path / "note.md"
        md_file.write_bytes(text.encode("latin-1"))
        conv = parse_md_file(md_file)
        assert len(conv.turns) >= 1

    def test_utf8_bom_still_works(self, tmp_path: Path):
        """UTF-8 with BOM should still work (regression check)."""
        md_file = tmp_path / "bom.md"
        md_file.write_bytes(b"\xef\xbb\xbf## BOM Content\n\nTest\n")
        conv = parse_md_file(md_file)
        assert len(conv.turns) >= 1


# ── Real sample files ────────────────────────────────────────────────


@pytest.mark.skipif(
    not MD_SAMPLES_DIR.exists(),
    reason="MD sample files not available",
)
class TestRealSamples:
    def test_parse_claude_sample(self):
        conv = parse_md_file(MD_SAMPLES_DIR / "claude.md")
        assert conv.meta.platform == "claude"
        assert conv.meta.url is not None
        assert "claude.ai" in conv.meta.url
        assert len(conv.turns) >= 3  # Multiple LLM 응답 sections

    def test_parse_chatgpt_sample(self):
        conv = parse_md_file(MD_SAMPLES_DIR / "chatgpt.md")
        assert conv.meta.platform == "chatgpt"
        assert conv.meta.url is not None
        assert len(conv.turns) >= 1

    def test_parse_gemini_sample(self):
        conv = parse_md_file(MD_SAMPLES_DIR / "gemini.md")
        assert conv.meta.platform == "gemini"
        assert conv.meta.url is not None
        assert len(conv.turns) >= 1

    def test_parse_grok_sample(self):
        conv = parse_md_file(MD_SAMPLES_DIR / "grok.md")
        assert conv.meta.platform == "grok"
        assert conv.meta.url is not None
        assert len(conv.turns) >= 1

    def test_parse_perplexity_sample(self):
        conv = parse_md_file(MD_SAMPLES_DIR / "perplexity.md")
        assert conv.meta.platform == "perplexity"
        assert conv.meta.url is not None
        assert len(conv.turns) >= 1

    def test_all_samples_parse(self):
        """All 5 platform samples should parse without error."""
        for name in ("claude", "chatgpt", "gemini", "grok", "perplexity"):
            conv = parse_md_file(MD_SAMPLES_DIR / f"{name}.md")
            assert conv.meta.platform == name
            assert conv.turn_count >= 1
