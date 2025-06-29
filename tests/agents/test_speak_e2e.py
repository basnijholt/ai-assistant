"""End-to-end tests for the speak agent with simplified mocks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_cli.agents.speak import async_main
from tests.mocks.audio import MockPyAudio
from tests.mocks.wyoming import MockTTSClient


@pytest.mark.asyncio
@patch("agent_cli.tts.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_e2e(
    mock_pyaudio_class: MagicMock,
    mock_async_client_class: MagicMock,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test end-to-end speech synthesis with simplified mocks."""
    # Setup mock PyAudio
    mock_pyaudio_class.return_value = MockPyAudio(mock_pyaudio_device_info)

    # Setup mock Wyoming client
    mock_tts_client = MockTTSClient(b"fake audio data")
    mock_async_client_class.from_uri.return_value.__aenter__.return_value = mock_tts_client

    await async_main(
        quiet=True,
        console=None,
        text="Hello, world!",
        tts_server_ip="mock-host",
        tts_server_port=10200,
        voice_name=None,
        tts_language=None,
        speaker=None,
        output_device_index=None,
        output_device_name=None,
        list_output_devices_flag=False,
        save_file=None,
    )

    # Verify that the audio was "played"
    assert mock_pyaudio_class.return_value.streams[0].get_written_data()
