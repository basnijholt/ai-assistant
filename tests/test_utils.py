"""Tests for the utility functions."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from agent_cli import utils


@pytest.mark.parametrize(
    ("td", "expected"),
    [
        (timedelta(seconds=5), "5 seconds ago"),
        (timedelta(minutes=5), "5 minutes ago"),
        (timedelta(hours=5), "5 hours ago"),
        (timedelta(days=5), "5 days ago"),
    ],
)
def test_format_timedelta_to_ago(td: timedelta, expected: str) -> None:
    """Test the format_timedelta_to_ago function."""
    assert utils.format_timedelta_to_ago(td) == expected


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


def test_print_device_index() -> None:
    """Test the print_device_index function."""
    console = MagicMock()
    utils.print_device_index(console, 1, "mock_device")
    console.print.assert_called_once()


def test_print_input_panel() -> None:
    """Test the print_input_panel function."""
    console = MagicMock()
    utils.print_input_panel(console, "hello")
    console.print.assert_called_once()


def test_print_output_panel() -> None:
    """Test the print_output_panel function."""
    console = MagicMock()
    utils.print_output_panel(console, "hello")
    console.print.assert_called_once()


def test_print_status_message() -> None:
    """Test the print_status_message function."""
    console = MagicMock()
    utils.print_status_message(console, "hello")
    console.print.assert_called_once()


def test_print_error_message() -> None:
    """Test the print_error_message function."""
    console = MagicMock()
    utils.print_error_message(console, "hello", "world")
    assert console.print.call_count == 2
