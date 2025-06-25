"""Tests for the transcribe agent."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assistant.agents import transcribe


@pytest.mark.asyncio
@patch("ai_assistant.agents.transcribe.asr")
@patch("ai_assistant.agents.transcribe.pyperclip")
@patch("ai_assistant.agents.transcribe.AsyncClient")
async def test_transcribe_main(
    mock_async_client_class: MagicMock,
    mock_pyperclip: MagicMock,
    mock_asr: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the main function of the transcribe agent."""
    mock_args = MagicMock()
    mock_args.list_devices = False
    mock_args.clipboard = True
    mock_args.asr_server_ip = "localhost"
    mock_args.asr_server_port = 12345
    mock_args.device_index = None

    # This is the client that is used in the 'async with' statement
    mock_async_client_instance = AsyncMock()
    # This is the class that is called
    mock_async_client_class.from_uri.return_value = mock_async_client_instance

    # Mock the context manager for the audio stream
    mock_pyaudio_stream = MagicMock()
    mock_asr.open_pyaudio_stream.return_value.__enter__.return_value = mock_pyaudio_stream

    # Mock the two async functions that are gathered
    mock_asr.send_audio = AsyncMock(return_value=None)
    mock_asr.receive_text = AsyncMock(return_value="hello world")

    # The function we are testing
    with caplog.at_level(logging.INFO):
        await transcribe.run_transcription(
            args=mock_args,
            logger=logging.getLogger(),
            p=MagicMock(),
            console=None,
        )

    # Assertions
    assert "Received transcript: hello world" in caplog.text
    assert "Copied transcript to clipboard." in caplog.text
    mock_pyperclip.copy.assert_called_once_with("hello world")
    mock_asr.send_audio.assert_awaited_once()
    mock_asr.receive_text.assert_awaited_once()
