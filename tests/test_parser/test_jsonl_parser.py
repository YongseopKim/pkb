"""Tests for JSONL parser."""

from pathlib import Path

import pytest

from pkb.parser.exceptions import MetaLineError, ParseError, TurnParseError
from pkb.parser.jsonl_parser import parse_jsonl_file, parse_jsonl_string


class TestParseJsonlString:
    def test_basic_parsing(self):
        data = (
            '{"_meta":true,"platform":"claude","url":"https://claude.ai/chat/abc",'
            '"exported_at":"2026-02-21T06:02:42.230Z","title":"Test"}\n'
            '{"role":"user","content":"Hello","timestamp":"2026-02-21T06:02:42.232Z"}\n'
            '{"role":"assistant","content":"Hi","timestamp":"2026-02-21T06:02:42.237Z"}\n'
        )
        conv = parse_jsonl_string(data)
        assert conv.meta.platform == "claude"
        assert conv.meta.title == "Test"
        assert len(conv.turns) == 2
        assert conv.turns[0].role == "user"
        assert conv.turns[1].role == "assistant"

    def test_meta_extraction(self):
        data = (
            '{"_meta":true,"platform":"chatgpt","url":"https://chatgpt.com/c/abc",'
            '"exported_at":"2026-02-21T06:02:37.664Z","title":"사이드 프로젝트: PKB"}\n'
            '{"role":"user","content":"Q","timestamp":"2026-02-21T06:02:37.666Z"}\n'
        )
        conv = parse_jsonl_string(data)
        assert conv.meta.platform == "chatgpt"
        assert "PKB" in conv.meta.title

    def test_consecutive_assistant_turns(self):
        """Claude-style: thinking + response as separate assistant turns."""
        data = (
            '{"_meta":true,"platform":"claude","url":"https://claude.ai/chat/abc",'
            '"exported_at":"2026-02-21T06:00:00.000Z","title":"T"}\n'
            '{"role":"user","content":"Q","timestamp":"2026-02-21T06:00:01.000Z"}\n'
            '{"role":"assistant","content":"thinking...","timestamp":"2026-02-21T06:00:02.000Z"}\n'
            '{"role":"assistant","content":"actual answer",'
            '"timestamp":"2026-02-21T06:00:03.000Z"}\n'
        )
        conv = parse_jsonl_string(data)
        assert len(conv.turns) == 3
        assert conv.turns[1].role == "assistant"
        assert conv.turns[2].role == "assistant"

    def test_multiple_assistant_blocks(self):
        """Perplexity-style: 1 user + multiple assistant blocks."""
        data = (
            '{"_meta":true,"platform":"perplexity","url":"https://perplexity.ai/abc",'
            '"exported_at":"2026-02-21T06:00:00.000Z","title":"T"}\n'
            '{"role":"user","content":"Q","timestamp":"2026-02-21T06:00:01.000Z"}\n'
            '{"role":"assistant","content":"A1","timestamp":"2026-02-21T06:00:02.000Z"}\n'
            '{"role":"assistant","content":"A2","timestamp":"2026-02-21T06:00:03.000Z"}\n'
            '{"role":"assistant","content":"A3","timestamp":"2026-02-21T06:00:04.000Z"}\n'
        )
        conv = parse_jsonl_string(data)
        assert len(conv.turns) == 4
        assert conv.first_user_message == "Q"

    def test_empty_content_raises(self):
        with pytest.raises(ParseError):
            parse_jsonl_string("")

    def test_no_meta_line_raises(self):
        data = '{"role":"user","content":"Q","timestamp":"2026-02-21T06:00:00.000Z"}\n'
        with pytest.raises(MetaLineError):
            parse_jsonl_string(data)

    def test_invalid_json_raises(self):
        data = "not json at all\n"
        with pytest.raises(MetaLineError):
            parse_jsonl_string(data)

    def test_missing_role_in_turn_raises(self):
        data = (
            '{"_meta":true,"platform":"claude","url":"x","exported_at":"2026-02-21T06:00:00.000Z","title":"T"}\n'
            '{"content":"no role","timestamp":"2026-02-21T06:00:01.000Z"}\n'
        )
        with pytest.raises(TurnParseError) as exc_info:
            parse_jsonl_string(data)
        assert exc_info.value.line_number == 2

    def test_turn_parse_error_has_line_number(self):
        data = (
            '{"_meta":true,"platform":"claude","url":"x","exported_at":"2026-02-21T06:00:00.000Z","title":"T"}\n'
            '{"role":"user","content":"ok","timestamp":"2026-02-21T06:00:01.000Z"}\n'
            'bad json line\n'
        )
        with pytest.raises(TurnParseError) as exc_info:
            parse_jsonl_string(data)
        assert exc_info.value.line_number == 3


class TestParseJsonlFile:
    def test_parse_all_sample_files(self, samples_dir: Path):
        """All 5 sample JSONL files must parse successfully."""
        expected_platforms = {"chatgpt", "claude", "gemini", "grok", "perplexity"}
        parsed_platforms = set()

        for jsonl_file in sorted(samples_dir.glob("*.jsonl")):
            conv = parse_jsonl_file(jsonl_file)
            assert conv.meta.platform in expected_platforms
            assert len(conv.turns) > 0
            assert conv.first_user_message is not None
            parsed_platforms.add(conv.meta.platform)

        assert parsed_platforms == expected_platforms

    def test_nonexistent_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_jsonl_file(tmp_path / "nonexistent.jsonl")

    def test_parse_chatgpt_sample(self, samples_dir: Path):
        conv = parse_jsonl_file(samples_dir / "chatgpt.jsonl")
        assert conv.meta.platform == "chatgpt"
        assert conv.turn_count > 0

    def test_parse_claude_sample(self, samples_dir: Path):
        """Claude has consecutive assistant turns — must not fail."""
        conv = parse_jsonl_file(samples_dir / "claude.jsonl")
        assert conv.meta.platform == "claude"
        # Claude sample has consecutive assistant turns
        roles = [t.role for t in conv.turns]
        # Check there are indeed consecutive assistants
        has_consecutive = any(
            roles[i] == "assistant" and roles[i + 1] == "assistant"
            for i in range(len(roles) - 1)
        )
        assert has_consecutive

    def test_parse_perplexity_sample(self, samples_dir: Path):
        """Perplexity has multiple assistant blocks."""
        conv = parse_jsonl_file(samples_dir / "perplexity.jsonl")
        assert conv.meta.platform == "perplexity"
        assert conv.turn_count > 0
