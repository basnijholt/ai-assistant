"""Tests for the fix_my_text agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from ollama import ResponseError
from rich.console import Console

from ai_assistant.agents import fix_my_text


@patch("ai_assistant.agents.fix_my_text.cli.get_base_parser")
@patch("ai_assistant.agents.fix_my_text.get_clipboard_text", return_value="hello")
@patch("ai_assistant.agents.fix_my_text.process_text", return_value=("hello world", 0.1))
@patch("ai_assistant.agents.fix_my_text._display_result")
def test_fix_my_text_main(
    mock_display_result: MagicMock,
    mock_process_text: MagicMock,
    mock_get_clipboard: MagicMock,
    mock_get_parser: MagicMock,
) -> None:
    """Test the main function of the fix_my_text agent."""
    mock_parser = MagicMock()
    mock_parser.parse_args.return_value.text = None
    mock_parser.parse_args.return_value.quiet = False
    mock_parser.parse_args.return_value.model = "test-model"
    mock_parser.parse_args.return_value.log_file = None  # Ensure log_file is None
    mock_get_parser.return_value = mock_parser

    with patch("sys.argv", ["fix_my_text", "--model", "test-model"]):
        fix_my_text.main()

    mock_get_clipboard.assert_called_once()
    mock_process_text.assert_called_once_with("hello", "test-model")
    mock_display_result.assert_called_once()
    args, kwargs = mock_display_result.call_args
    assert args[0] == "hello world"
    assert args[1] == "hello"
    assert args[2] == 0.1
    assert not kwargs["simple_output"]
    assert isinstance(kwargs["console"], Console)


@patch("ai_assistant.agents.fix_my_text.cli.get_base_parser")
@patch("ai_assistant.agents.fix_my_text.get_clipboard_text", return_value="hello")
@patch("ai_assistant.agents.fix_my_text.process_text", return_value=("hello world", 0.1))
@patch("builtins.print")
def test_fix_my_text_main_quiet(
    mock_print: MagicMock,
    mock_process_text: MagicMock,
    mock_get_parser: MagicMock,
) -> None:
    """Test the main function in quiet mode."""
    mock_parser = MagicMock()
    mock_parser.parse_args.return_value.text = None
    mock_parser.parse_args.return_value.quiet = True
    mock_parser.parse_args.return_value.model = "test-model"
    mock_parser.parse_args.return_value.log_file = None
    mock_get_parser.return_value = mock_parser

    with patch("sys.argv", ["fix_my_text", "--quiet"]):
        fix_my_text.main()

    mock_process_text.assert_called_once_with("hello", "test-model")
    mock_print.assert_called_with("hello world")


@patch("ai_assistant.agents.fix_my_text.cli.get_base_parser")
@patch("ai_assistant.agents.fix_my_text.get_clipboard_text", return_value="hello")
@patch(
    "ai_assistant.agents.fix_my_text.process_text",
    side_effect=ResponseError("Test error"),
)
@patch("builtins.print")
def test_fix_my_text_main_error(
    mock_print: MagicMock,
    mock_process_text: MagicMock,
    mock_get_parser: MagicMock,
) -> None:
    """Test the main function with an error."""
    mock_parser = MagicMock()
    mock_parser.parse_args.return_value.text = None
    mock_parser.parse_args.return_value.quiet = True
    mock_parser.parse_args.return_value.model = "test-model"
    mock_parser.parse_args.return_value.log_file = None
    mock_get_parser.return_value = mock_parser

    with patch("sys.argv", ["fix_my_text", "--quiet"]), pytest.raises(SystemExit):
        fix_my_text.main()

    mock_process_text.assert_called_once()
    mock_print.assert_called_with("❌ Test error (status code: -1)")
