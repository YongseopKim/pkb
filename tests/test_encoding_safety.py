"""Tests for UTF-8 encoding safety across PKB file reads.

Verifies that files with a few broken UTF-8 bytes are handled gracefully
via errors='replace' instead of raising UnicodeDecodeError.
"""

from pathlib import Path


def _write_broken_utf8(path: Path) -> None:
    """Write a mostly-valid UTF-8 file with a few broken bytes."""
    # Valid UTF-8 Korean + broken byte sequence in the middle
    # "성장" (EC 84 B1 EC 9E A5) then broken: EB A5 2A (incomplete 3-byte seq)
    content = (
        b"---\n"
        b"id: 20260101-test-abc1\n"
        b"question: \xed\x85\x8c\xec\x8a\xa4\xed\x8a\xb8\n"  # "테스트"
        b"summary: \xec\x9a\x94\xec\x95\xbd\n"  # "요약"
        b"domains:\n"
        b"  - dev\n"
        b"topics:\n"
        b"  - python\n"
        b"---\n"
        b"\n# Content\n\xec\x84\xb1\xec\x9e\xa5\xeb\xa5\x2a\n"  # broken byte
    )
    path.write_bytes(content)


class TestFrontmatterParserEncodingSafety:
    """frontmatter_parser가 깨진 UTF-8 파일을 처리할 수 있는지 테스트."""

    def test_parse_frontmatter_broken_utf8(self, tmp_path):
        """깨진 UTF-8 바이트가 있는 파일에서 frontmatter 파싱 성공."""
        from pkb.generator.frontmatter_parser import parse_frontmatter

        md = tmp_path / "broken.md"
        _write_broken_utf8(md)
        result = parse_frontmatter(md)
        assert result["id"] == "20260101-test-abc1"
        assert result["domains"] == ["dev"]

    def test_parse_md_body_broken_utf8(self, tmp_path):
        """깨진 UTF-8 바이트가 있는 파일에서 body 추출 성공."""
        from pkb.generator.frontmatter_parser import parse_md_body

        md = tmp_path / "broken.md"
        _write_broken_utf8(md)
        body = parse_md_body(md)
        assert "# Content" in body


class TestReindexEncodingSafety:
    """reindex에서 깨진 UTF-8 MD 파일을 읽을 수 있는지 테스트."""

    def test_read_text_with_replace_on_broken_utf8(self, tmp_path):
        """reindex._rechunk_bundle에서 사용되는 read_text가 깨진 파일을 처리."""
        md = tmp_path / "platform.md"
        # Frontmatter 없는 MD (ValueError 발생 시 fallback read_text 경로)
        md.write_bytes(b"# Hello\n\xec\x84\xb1\xec\x9e\xa5\xeb\xa5\x2a world\n")

        # errors='replace' 없이는 UnicodeDecodeError
        text = md.read_text(encoding="utf-8", errors="replace")
        assert "# Hello" in text
        assert "world" in text


class TestReembedEncodingSafety:
    """reembed에서 깨진 UTF-8 MD 파일을 읽을 수 있는지 테스트."""

    def test_read_text_with_replace_on_broken_utf8(self, tmp_path):
        """reembed에서 사용되는 read_text가 깨진 파일을 처리."""
        md = tmp_path / "platform.md"
        md.write_bytes(b"# Hello\n\xec\x84\xb1\xec\x9e\xa5\xeb\xa5\x2a world\n")

        text = md.read_text(encoding="utf-8", errors="replace")
        assert "# Hello" in text
        assert "\ufffd" in text  # replacement character present


class TestReadTextWithFallbackEncodingSafety:
    """read_text_with_fallback가 깨진 UTF-8 파일을 올바르게 처리하는지 테스트."""

    def test_mostly_utf8_with_broken_bytes(self, tmp_path):
        """거의 UTF-8이지만 일부 바이트가 깨진 파일 → UTF-8 replace로 처리."""
        from pkb.parser import read_text_with_fallback

        md = tmp_path / "broken.md"
        # Valid Korean UTF-8 + one broken byte in the middle
        md.write_bytes(
            b"\xed\x85\x8c\xec\x8a\xa4\xed\x8a\xb8 "  # "테스트 "
            b"\xeb\xa5\x2a "  # broken: EB A5 then 2A (*) instead of 80-BF
            b"\xed\x8c\x8c\xec\x9d\xbc"  # "파일"
        )
        text = read_text_with_fallback(md)
        assert "테스트" in text
        assert "파일" in text
        # Should NOT fall through to latin-1 which would garble everything
        assert "\ufffd" in text  # replacement character for the broken byte

    def test_pure_utf8_still_works(self, tmp_path):
        """순수 UTF-8 파일은 여전히 정상 처리."""
        from pkb.parser import read_text_with_fallback

        md = tmp_path / "clean.md"
        md.write_text("안녕하세요 테스트입니다", encoding="utf-8")
        text = read_text_with_fallback(md)
        assert text == "안녕하세요 테스트입니다"

    def test_cp949_still_works(self, tmp_path):
        """CP949 파일도 여전히 정상 처리."""
        from pkb.parser import read_text_with_fallback

        md = tmp_path / "legacy.md"
        md.write_bytes("한글 테스트".encode("cp949"))
        text = read_text_with_fallback(md)
        assert "한글" in text
        assert "테스트" in text
