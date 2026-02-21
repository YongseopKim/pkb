"""Tests for prompt template loading."""

from pkb.generator.prompts import load_prompt, render_prompt


class TestLoadPrompt:
    def test_load_response_meta(self):
        text = load_prompt("response_meta")
        assert isinstance(text, str)
        assert len(text) > 50
        assert "{" in text  # Has template variables

    def test_load_bundle_meta(self):
        text = load_prompt("bundle_meta")
        assert isinstance(text, str)
        assert len(text) > 50

    def test_load_nonexistent_raises(self):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt")


class TestRenderPrompt:
    def test_render_with_variables(self):
        template = "Platform: {platform}\nContent: {content}"
        rendered = render_prompt(template, platform="claude", content="테스트 내용")
        assert "Platform: claude" in rendered
        assert "Content: 테스트 내용" in rendered

    def test_render_preserves_unmatched(self):
        template = "Hello {name}, your {role} is ready"
        rendered = render_prompt(template, name="PKB")
        assert "Hello PKB" in rendered
