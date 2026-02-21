"""Tests for directory parser."""

import json
from pathlib import Path

import pytest

from pkb.parser.directory import (
    find_input_files,
    find_jsonl_files,
    parse_directory,
    parse_file,
)
from pkb.parser.exceptions import ParseError


def _write_sample_jsonl(path: Path, platform: str, exported_at: str) -> None:
    """Helper to write a minimal JSONL file."""
    lines = [
        json.dumps({
            "_meta": True,
            "platform": platform,
            "url": f"https://{platform}.example.com",
            "exported_at": exported_at,
            "title": f"Test {platform}",
        }),
        json.dumps({
            "role": "user",
            "content": "Hello",
            "timestamp": exported_at,
        }),
        json.dumps({
            "role": "assistant",
            "content": "Hi there",
            "timestamp": exported_at,
        }),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestFindJsonlFiles:
    def test_finds_jsonl_files(self, tmp_path: Path):
        _write_sample_jsonl(tmp_path / "a.jsonl", "claude", "2026-02-21T06:00:00.000Z")
        _write_sample_jsonl(tmp_path / "b.jsonl", "chatgpt", "2026-02-21T06:00:00.000Z")
        files = find_jsonl_files(tmp_path)
        assert len(files) == 2

    def test_ignores_non_jsonl(self, tmp_path: Path):
        _write_sample_jsonl(tmp_path / "a.jsonl", "claude", "2026-02-21T06:00:00.000Z")
        (tmp_path / "readme.md").write_text("# Hello")
        (tmp_path / "notes.txt").write_text("notes")
        files = find_jsonl_files(tmp_path)
        assert len(files) == 1

    def test_empty_directory(self, tmp_path: Path):
        files = find_jsonl_files(tmp_path)
        assert files == []

    def test_nonexistent_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            find_jsonl_files(tmp_path / "nonexistent")


class TestParseDirectory:
    def test_parses_all_files(self, tmp_path: Path):
        _write_sample_jsonl(tmp_path / "claude.jsonl", "claude", "2026-02-21T07:00:00.000Z")
        _write_sample_jsonl(tmp_path / "chatgpt.jsonl", "chatgpt", "2026-02-21T06:00:00.000Z")
        conversations = parse_directory(tmp_path)
        assert len(conversations) == 2

    def test_sorted_by_exported_at(self, tmp_path: Path):
        _write_sample_jsonl(tmp_path / "later.jsonl", "claude", "2026-02-21T08:00:00.000Z")
        _write_sample_jsonl(tmp_path / "earlier.jsonl", "chatgpt", "2026-02-21T06:00:00.000Z")
        conversations = parse_directory(tmp_path)
        assert conversations[0].meta.platform == "chatgpt"  # earlier
        assert conversations[1].meta.platform == "claude"  # later

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        conversations = parse_directory(tmp_path)
        assert conversations == []

    def test_with_real_samples(self, samples_dir: Path):
        """Parse the real exporter-examples directory."""
        conversations = parse_directory(samples_dir)
        assert len(conversations) == 5
        platforms = {c.meta.platform for c in conversations}
        assert platforms == {"chatgpt", "claude", "gemini", "grok", "perplexity"}


def _write_sample_md(path: Path, platform: str, content: str = "Content") -> None:
    """Helper to write a minimal MD file."""
    path.write_text(
        f"# [{platform}](https://{platform}.example.com/chat/abc)\n\n"
        f"---\n---\n\n## LLM 응답 1\n\n{content}\n",
        encoding="utf-8",
    )


class TestFindInputFiles:
    def test_finds_both_jsonl_and_md(self, tmp_path: Path):
        _write_sample_jsonl(tmp_path / "a.jsonl", "claude", "2026-02-21T06:00:00.000Z")
        _write_sample_md(tmp_path / "gemini.md", "gemini")
        files = find_input_files(tmp_path)
        assert len(files) == 2
        extensions = {f.suffix for f in files}
        assert extensions == {".jsonl", ".md"}

    def test_ignores_unsupported(self, tmp_path: Path):
        _write_sample_jsonl(tmp_path / "a.jsonl", "claude", "2026-02-21T06:00:00.000Z")
        (tmp_path / "notes.txt").write_text("notes")
        (tmp_path / "data.csv").write_text("a,b,c")
        files = find_input_files(tmp_path)
        assert len(files) == 1

    def test_empty_directory(self, tmp_path: Path):
        files = find_input_files(tmp_path)
        assert files == []

    def test_nonexistent_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            find_input_files(tmp_path / "nonexistent")


class TestFindInputFilesRecursive:
    """Tests for recursive file discovery with .done/ exclusion."""

    def test_finds_files_in_subdirectories(self, tmp_path: Path):
        """서브디렉토리 내 파일도 찾아야 함."""
        from pkb.parser.directory import find_input_files_recursive

        sub = tmp_path / "PKB"
        sub.mkdir()
        _write_sample_jsonl(sub / "claude.jsonl", "claude", "2026-02-21T06:00:00.000Z")
        _write_sample_md(sub / "chatgpt.md", "chatgpt")

        files = find_input_files_recursive(tmp_path)
        assert len(files) == 2
        assert all(f.is_relative_to(sub) for f in files)

    def test_excludes_done_dir_recursive(self, tmp_path: Path):
        """재귀 탐색 시 .done/ 내부 파일은 제외."""
        from pkb.parser.directory import find_input_files_recursive

        (tmp_path / "test.jsonl").write_text("data")
        done = tmp_path / ".done"
        done.mkdir()
        (done / "old.jsonl").write_text("data")
        # Nested .done inside subdirectory
        sub = tmp_path / "sub"
        sub.mkdir()
        nested_done = sub / ".done"
        nested_done.mkdir()
        (nested_done / "old.md").write_text("data")

        files = find_input_files_recursive(tmp_path)
        assert len(files) == 1
        assert files[0].name == "test.jsonl"

    def test_finds_flat_and_nested_files(self, tmp_path: Path):
        """직하 파일과 중첩 파일을 모두 찾아야 함."""
        from pkb.parser.directory import find_input_files_recursive

        _write_sample_jsonl(tmp_path / "top.jsonl", "claude", "2026-02-21T06:00:00.000Z")
        sub = tmp_path / "PKB"
        sub.mkdir()
        _write_sample_md(sub / "gemini.md", "gemini")
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        _write_sample_jsonl(deep / "deep.jsonl", "grok", "2026-02-21T06:00:00.000Z")

        files = find_input_files_recursive(tmp_path)
        assert len(files) == 3

    def test_empty_directory_recursive(self, tmp_path: Path):
        """빈 디렉토리는 빈 리스트 반환."""
        from pkb.parser.directory import find_input_files_recursive

        files = find_input_files_recursive(tmp_path)
        assert files == []

    def test_nonexistent_directory_recursive_raises(self, tmp_path: Path):
        """존재하지 않는 디렉토리는 FileNotFoundError."""
        from pkb.parser.directory import find_input_files_recursive

        with pytest.raises(FileNotFoundError):
            find_input_files_recursive(tmp_path / "nonexistent")


class TestParseFile:
    def test_dispatches_jsonl(self, tmp_path: Path):
        _write_sample_jsonl(tmp_path / "test.jsonl", "claude", "2026-02-21T06:00:00.000Z")
        conv = parse_file(tmp_path / "test.jsonl")
        assert conv.meta.platform == "claude"

    def test_dispatches_md(self, tmp_path: Path):
        _write_sample_md(tmp_path / "claude.md", "claude")
        conv = parse_file(tmp_path / "claude.md")
        assert len(conv.turns) >= 1

    def test_unsupported_extension_raises(self, tmp_path: Path):
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("hello")
        with pytest.raises(ParseError):
            parse_file(txt_file)


class TestParseDirectoryMixed:
    """Test parse_directory with both JSONL and MD files."""

    def test_parses_mixed_directory(self, tmp_path: Path):
        _write_sample_jsonl(tmp_path / "chat.jsonl", "claude", "2026-02-21T06:00:00.000Z")
        _write_sample_md(tmp_path / "gemini.md", "gemini")
        conversations = parse_directory(tmp_path)
        assert len(conversations) == 2
