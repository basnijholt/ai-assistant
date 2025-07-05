"""Tests for the voice assistant agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from agent_cli.cli import app

runner = CliRunner()


@patch("agent_cli.agents.voice_assistant.asr.transcribe_audio", new_callable=AsyncMock)
@patch("agent_cli.agents.voice_assistant.process_and_update_clipboard", new_callable=AsyncMock)
@patch("agent_cli.agents.voice_assistant.pyaudio_context")
@patch("agent_cli.agents.voice_assistant.get_clipboard_text")
@patch("agent_cli.agents.voice_assistant.signal_handling_context")
@patch("agent_cli.agents.voice_assistant.maybe_live")
def test_voice_assistant_agent(
    mock_maybe_live: MagicMock,
    mock_signal_handling: MagicMock,
    mock_get_clipboard_text: MagicMock,
    mock_pyaudio_context: MagicMock,
    mock_process_and_update_clipboard: AsyncMock,
    mock_transcribe_audio: AsyncMock,
) -> None:
    """Test the voice assistant agent."""
    mock_get_clipboard_text.return_value = "hello"
    mock_transcribe_audio.return_value = "world"
    mock_signal_handling.return_value.__aenter__.return_value = MagicMock()
    mock_maybe_live.return_value.__enter__.return_value = MagicMock()
    result = runner.invoke(app, ["voice-assistant", "--config", "missing.toml"])
    assert result.exit_code == 0, result.output
    mock_pyaudio_context.assert_called_once()
    mock_process_and_update_clipboard.assert_called_once()
    mock_transcribe_audio.assert_called_once()


@patch("agent_cli.agents.voice_assistant.process_manager.kill_process")
def test_voice_assistant_stop(mock_kill_process: MagicMock) -> None:
    """Test the --stop flag."""
    mock_kill_process.return_value = True
    result = runner.invoke(app, ["voice-assistant", "--stop"])
    assert result.exit_code == 0
    assert "Voice assistant stopped" in result.stdout
    mock_kill_process.assert_called_once_with("voice-assistant")


@patch("agent_cli.agents.voice_assistant.process_manager.kill_process")
def test_voice_assistant_stop_not_running(mock_kill_process: MagicMock) -> None:
    """Test the --stop flag when the process is not running."""
    mock_kill_process.return_value = False
    result = runner.invoke(app, ["voice-assistant", "--stop"])
    assert result.exit_code == 0
    assert "No voice assistant is running" in result.stdout


@patch("agent_cli.agents.voice_assistant.process_manager.is_process_running")
def test_voice_assistant_status_running(mock_is_process_running: MagicMock) -> None:
    """Test the --status flag when the process is running."""
    mock_is_process_running.return_value = True
    with patch(
        "agent_cli.agents.voice_assistant.process_manager.read_pid_file",
        return_value=123,
    ):
        result = runner.invoke(app, ["voice-assistant", "--status"])
    assert result.exit_code == 0
    assert "Voice assistant is running" in result.stdout


@patch("agent_cli.agents.voice_assistant.process_manager.is_process_running")
def test_voice_assistant_status_not_running(mock_is_process_running: MagicMock) -> None:
    """Test the --status flag when the process is not running."""
    mock_is_process_running.return_value = False
    result = runner.invoke(app, ["voice-assistant", "--status"])
    assert result.exit_code == 0
    assert "Voice assistant is not running" in result.stdout
