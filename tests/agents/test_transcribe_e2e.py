"""End-to-end tests for the transcribe agent with minimal mocking."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_cli.agents.transcribe import async_main
from tests.mocks.audio import MockPyAudio
from tests.mocks.wyoming import MockASRClient


@pytest.mark.asyncio
@patch("wyoming.client.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_transcribe_e2e(
    mock_pyaudio_class: MagicMock,
    mock_async_client_class: MagicMock,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test end-to-end transcription with simplified mocks."""
    # Setup mock PyAudio
    mock_pyaudio_class.return_value = MockPyAudio(mock_pyaudio_device_info)

    # Setup mock Wyoming client
    mock_asr_client = MockASRClient("This is a test transcription.")
    mock_async_client_class.from_uri.return_value = mock_asr_client

    await async_main(
        device_index=0,
        asr_server_ip="mock-host",
        asr_server_port=10300,
        clipboard=False,
        quiet=True,
        llm=False,
        model="",
        ollama_host="",
        console=None,
        p=mock_pyaudio_class.return_value,
    )
