"""Tests for the utility functions."""

from __future__ import annotations

from unittest.mock import patch

from agent_cli import utils


def test_get_clipboard_text() -> None:
    """Test reading from clipboard."""
    with patch("pyperclip.paste", return_value="hello world"):
        text = utils.get_clipboard_text(console=None)
        assert text == "hello world"


def test_get_clipboard_text_empty() -> None:
    """Test reading from an empty clipboard."""
    with patch("pyperclip.paste", return_value=""):
        text = utils.get_clipboard_text(console=None)
        assert text is None
