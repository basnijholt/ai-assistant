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
    mock_get_clipboard: MagicMock,
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

    mock_get_clipboard.assert_called_once()
    mock_process_text.assert_called_once_with("hello", "test-model")
    mock_print.assert_called_with("hello world")


@patch("ai_assistant.agents.fix_my_text.get_clipboard_text", return_value="hello")
@patch(
    "ai_assistant.agents.fix_my_text.process_text",
    side_effect=ResponseError("Test error"),
)
@patch("ai_assistant.agents.fix_my_text.cli.get_base_parser")
@patch("builtins.print")
def test_fix_my_text_main_error(
    mock_print: MagicMock,
    mock_get_parser: MagicMock,
    mock_process_text: MagicMock,  # noqa: ARG001
    mock_get_clipboard: MagicMock,  # noqa: ARG001
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

    mock_print.assert_called_with("❌ Test error (status code: -1)")


@patch("ai_assistant.agents.fix_my_text.build_agent")
def test_process_text(mock_build_agent: MagicMock) -> None:
    """Test the process_text function."""
    mock_agent = MagicMock()
    mock_agent.run.return_value.output = "corrected text"
    mock_build_agent.return_value = mock_agent

    result, _ = fix_my_text.process_text("original text", "test-model")

    assert result == "corrected text"
    mock_build_agent.assert_called_once_with(
        model="test-model",
        ollama_host=fix_my_text.OLLAMA_HOST,
    )
    mock_agent.run.assert_called_once_with(
        "original text",
        system_prompt=fix_my_text.SYSTEM_PROMPT,
        instructions=fix_my_text.AGENT_INSTRUCTIONS,
    )


@patch("ai_assistant.agents.fix_my_text.pyperclip.copy")
def test_display_result(mock_copy: MagicMock) -> None:
    """Test the _display_result function."""
    mock_console = MagicMock()
    fix_my_text._display_result(
        "corrected",
        "original",
        0.1,
        simple_output=False,
        console=mock_console,
    )
    mock_copy.assert_called_once_with("corrected")
    assert mock_console.print.call_count == 2


@patch("ai_assistant.agents.fix_my_text.pyperclip.copy")
@patch("builtins.print")
def test_display_result_quiet(mock_print: MagicMock, mock_copy: MagicMock) -> None:
    """Test the _display_result function in quiet mode."""
    fix_my_text._display_result(
        "corrected",
        "original",
        0.1,
        simple_output=True,
        console=None,
    )
    mock_copy.assert_called_once_with("corrected")
    mock_print.assert_called_once_with("corrected")

    # Reset mocks for the second test
    mock_copy.reset_mock()
    mock_print.reset_mock()

    # Test no correction needed
    fix_my_text._display_result(
        "original",
        "original",
        0.1,
        simple_output=True,
        console=None,
    )
    mock_copy.assert_called_once_with("original")
    mock_print.assert_called_once_with("✅ No correction needed.")


@patch("ai_assistant.agents.fix_my_text.get_clipboard_text", return_value="hello")
@patch(
    "ai_assistant.agents.fix_my_text.process_text",
    side_effect=ResponseError("Test error"),
)
@patch("ai_assistant.agents.fix_my_text.Console")
@patch("ai_assistant.agents.fix_my_text.cli.get_base_parser")
def test_fix_my_text_main_error_verbose(
    mock_get_parser: MagicMock,
    mock_console: MagicMock,
    mock_process_text: MagicMock,  # noqa: ARG001
    mock_get_clipboard: MagicMock,  # noqa: ARG001
) -> None:
    """Test the main function with an error in verbose mode."""
    mock_parser = MagicMock()
    mock_parser.parse_args.return_value.text = None
    mock_parser.parse_args.return_value.quiet = False
    mock_parser.parse_args.return_value.model = "test-model"
    mock_parser.parse_args.return_value.log_file = None
    mock_get_parser.return_value = mock_parser

    with pytest.raises(SystemExit):
        fix_my_text.main()

    assert mock_console().print.call_count > 0
