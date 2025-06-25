"""Tests for the autocorrect agent."""

from __future__ import annotations

import contextlib
import io
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from ai_assistant import config
from ai_assistant.agents import autocorrect


def test_system_prompt_and_instructions():
    """Test that the system prompt and instructions are properly defined."""
    assert autocorrect.SYSTEM_PROMPT
    assert "editor" in autocorrect.SYSTEM_PROMPT.lower()
    assert "correct" in autocorrect.SYSTEM_PROMPT.lower()

    assert autocorrect.AGENT_INSTRUCTIONS
    assert "grammar" in autocorrect.AGENT_INSTRUCTIONS.lower()
    assert "spelling" in autocorrect.AGENT_INSTRUCTIONS.lower()


def test_display_result_quiet_mode():
    """Test the _display_result function in quiet mode with real output."""
    # Test normal correction
    with patch("ai_assistant.agents.autocorrect.pyperclip.copy") as mock_copy:
        output = io.StringIO()
        with redirect_stdout(output):
            autocorrect._display_result(
                "Hello world!",
                "hello world",
                0.1,
                simple_output=True,
                console=None,
            )

        assert output.getvalue().strip() == "Hello world!"
        mock_copy.assert_called_once_with("Hello world!")


def test_display_result_no_correction_needed():
    """Test the _display_result function when no correction is needed."""
    with patch("ai_assistant.agents.autocorrect.pyperclip.copy") as mock_copy:
        output = io.StringIO()
        with redirect_stdout(output):
            autocorrect._display_result(
                "Hello world!",
                "Hello world!",
                0.1,
                simple_output=True,
                console=None,
            )

        assert output.getvalue().strip() == "✅ No correction needed."
        mock_copy.assert_called_once_with("Hello world!")


def test_display_result_verbose_mode():
    """Test the _display_result function in verbose mode with real console output."""
    console = Console(file=io.StringIO(), width=80)

    with patch("ai_assistant.agents.autocorrect.pyperclip.copy") as mock_copy:
        autocorrect._display_result(
            "Hello world!",
            "hello world",
            0.25,
            simple_output=False,
            console=console,
        )

        output = console.file.getvalue()
        assert "Hello world!" in output
        assert "Corrected Text" in output
        assert "Success!" in output
        assert "0.25" in output  # Just check the number, not exact format
        assert "seconds" in output
        mock_copy.assert_called_once_with("Hello world!")


def test_display_original_text():
    """Test the display_original_text function."""
    console = Console(file=io.StringIO(), width=80)

    autocorrect.display_original_text("Test text here", console)

    output = console.file.getvalue()
    assert "Test text here" in output
    assert "Original Text" in output


def test_display_original_text_none_console():
    """Test display_original_text with None console (should not crash)."""
    # This should not raise an exception
    autocorrect.display_original_text("Test text", None)


@pytest.mark.asyncio
@patch("ai_assistant.agents.autocorrect.build_agent")
async def test_process_text_integration(mock_build_agent: MagicMock) -> None:
    """Test process_text with a more realistic mock setup."""
    # Create a mock agent that behaves more like the real thing
    mock_agent = MagicMock()
    mock_result = MagicMock()
    mock_result.output = "This is corrected text."
    mock_agent.run = AsyncMock(return_value=mock_result)
    mock_build_agent.return_value = mock_agent

    # Test the function
    result, elapsed = await autocorrect.process_text(
        "this is text",
        "test-model",
        "http://localhost:11434",
    )

    # Verify the result
    assert result == "This is corrected text."
    assert isinstance(elapsed, float)
    assert elapsed >= 0

    # Verify the agent was called correctly
    mock_build_agent.assert_called_once_with(
        model="test-model",
        ollama_host="http://localhost:11434",
    )
    mock_agent.run.assert_called_once_with(
        "this is text",
        system_prompt=autocorrect.SYSTEM_PROMPT,
        instructions=autocorrect.AGENT_INSTRUCTIONS,
    )


def test_configuration_constants():
    """Test that configuration constants are properly set."""
    # Test that OLLAMA_HOST has a reasonable value (could be localhost or custom)
    assert config.OLLAMA_HOST
    assert config.OLLAMA_HOST.startswith("http")  # Should be a valid URL

    # Test that DEFAULT_MODEL is set
    assert config.DEFAULT_MODEL
    assert isinstance(config.DEFAULT_MODEL, str)


# Keep one minimal integration test for the main function to ensure it doesn't crash
@patch("ai_assistant.agents.autocorrect.process_text")
@patch("ai_assistant.agents.autocorrect.get_clipboard_text")
def test_main_basic_integration(
    mock_get_clipboard: MagicMock,
    mock_process_text: MagicMock,
) -> None:
    """Basic integration test for main function - minimal mocking."""
    mock_get_clipboard.return_value = "test text"
    mock_process_text.return_value = ("Test text.", 0.1)

    # Test with direct text input (no clipboard needed)
    with (
        patch("sys.argv", ["autocorrect", "--quiet", "hello world"]),
        patch("ai_assistant.agents.autocorrect.pyperclip.copy"),
        redirect_stdout(io.StringIO()),
        contextlib.suppress(SystemExit),
    ):
        # This should not crash
        autocorrect.main()


@patch("ai_assistant.agents.autocorrect.get_clipboard_text")
def test_main_with_text_argument(mock_get_clipboard: MagicMock) -> None:
    """Test main function with text provided as argument."""
    mock_get_clipboard.return_value = "fallback text"

    with patch("ai_assistant.agents.autocorrect.process_text") as mock_process:
        mock_process.return_value = ("Corrected text.", 0.1)

        with (
            patch("sys.argv", ["autocorrect", "--quiet", "input text"]),
            patch("ai_assistant.agents.autocorrect.pyperclip.copy"),
            redirect_stdout(io.StringIO()),
            contextlib.suppress(SystemExit),
        ):
            autocorrect.main()

    # Should not have called clipboard since text was provided
    mock_get_clipboard.assert_not_called()
    # Should have processed the provided text
    mock_process.assert_called_with(
        "input text",
        config.DEFAULT_MODEL,
        config.OLLAMA_HOST,
    )
