"""Tests for the transcribe agent."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assistant.agents import transcribe


@pytest.mark.asyncio
@patch("ai_assistant.agents.transcribe.asr")
@patch("ai_assistant.agents.transcribe.pyperclip")
async def test_transcribe_main(
    mock_pyperclip: MagicMock,
    mock_asr: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the main function of the transcribe agent."""
    # Mock the pyaudio context manager
    mock_pyaudio_instance = MagicMock()
    mock_asr.pyaudio_context.return_value.__enter__.return_value = mock_pyaudio_instance

    # Mock the unified transcribe_audio function
    mock_asr.transcribe_audio = AsyncMock(return_value="hello world")

    # The function we are testing
    with caplog.at_level(logging.INFO):
        await transcribe.async_main(
            device_index=None,
            asr_server_ip="localhost",
            asr_server_port=12345,
            clipboard=True,
            quiet=True,  # To avoid console output in tests
            list_devices=False,
        )

    # Assertions
    assert "Copied transcript to clipboard." in caplog.text
    mock_pyperclip.copy.assert_called_once_with("hello world")
    mock_asr.transcribe_audio.assert_awaited_once()

    # Verify the correct arguments were passed to transcribe_audio
    call_args = mock_asr.transcribe_audio.call_args
    assert call_args.kwargs["asr_server_ip"] == "localhost"
    assert call_args.kwargs["asr_server_port"] == 12345
    assert call_args.kwargs["device_index"] is None
    assert call_args.kwargs["send_transcribe_event"] is False
