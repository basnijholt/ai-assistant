"""End-to-end tests for the transcribe agent with minimal mocking."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from agent_cli.agents.transcribe import async_main
from tests.mocks.audio import MockPyAudio
from tests.mocks.wyoming import MockASRClient

if TYPE_CHECKING:
    from rich.console import Console


@pytest.mark.asyncio
@patch("agent_cli.agents.transcribe.signal_handling_context")
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_transcribe_e2e(
    mock_pyaudio_class: MagicMock,
    mock_async_client_class: MagicMock,
    mock_signal_handling_context: MagicMock,
    mock_pyaudio_device_info: list[dict],
    mock_console: Console,
) -> None:
    """Test end-to-end transcription with simplified mocks."""
    # Setup mock PyAudio
    mock_pyaudio_instance = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio_instance

    # Setup mock Wyoming client
    transcript_text = "This is a test transcription."
    mock_asr_client = MockASRClient(transcript_text)
    mock_async_client_class.from_uri.return_value = mock_asr_client

    # Setup stop event
    stop_event = asyncio.Event()
    mock_signal_handling_context.return_value.__enter__.return_value = stop_event
    asyncio.get_event_loop().call_later(0.1, stop_event.set)

    await async_main(
        device_index=0,
        asr_server_ip="mock-host",
        asr_server_port=10300,
        clipboard=False,
        quiet=False,
        llm=False,
        model="",
        ollama_host="",
        console=mock_console,
        p=mock_pyaudio_instance,
    )

    # Assert that the final transcript is in the console output
    output = mock_console.file.getvalue()
    assert transcript_text in output

    # Ensure the mock client was used
    mock_async_client_class.from_uri.assert_called_once()
