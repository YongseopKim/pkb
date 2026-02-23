"""Tests for ChatEngine conversation modes."""

from unittest.mock import MagicMock

import pytest


class TestChatModes:
    def test_explorer_mode_is_default(self):
        from pkb.chat.engine import ChatEngine

        engine = ChatEngine(
            search_engine=MagicMock(),
            router=MagicMock(),
        )
        assert engine._mode == "explorer"

    def test_analyst_mode_accepted(self):
        from pkb.chat.engine import ChatEngine

        engine = ChatEngine(
            search_engine=MagicMock(),
            router=MagicMock(),
            mode="analyst",
        )
        assert engine._mode == "analyst"

    def test_writer_mode_accepted(self):
        from pkb.chat.engine import ChatEngine

        engine = ChatEngine(
            search_engine=MagicMock(),
            router=MagicMock(),
            mode="writer",
        )
        assert engine._mode == "writer"

    def test_invalid_mode_raises(self):
        from pkb.chat.engine import ChatEngine

        with pytest.raises(ValueError):
            ChatEngine(
                search_engine=MagicMock(),
                router=MagicMock(),
                mode="invalid",
            )

    def test_valid_modes_constant(self):
        from pkb.chat.engine import VALID_MODES

        assert "explorer" in VALID_MODES
        assert "analyst" in VALID_MODES
        assert "writer" in VALID_MODES
