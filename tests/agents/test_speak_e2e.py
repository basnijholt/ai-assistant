"""End-to-end tests for the speak agent with simplified mocks."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from agent_cli.agents._config import FileConfig, GeneralConfig, TTSConfig
from agent_cli.agents.speak import async_main
from tests.mocks.audio import MockPyAudio
from tests.mocks.wyoming import MockTTSClient

if TYPE_CHECKING:
    from rich.console import Console


@pytest.mark.asyncio
@patch("agent_cli.tts.pyaudio_context")
@patch("agent_cli.agents.speak.pyaudio_context")
@patch("agent_cli.tts.AsyncClient")
async def test_speak_e2e(
    mock_async_client_class: MagicMock,
    mock_pyaudio_context_speak: MagicMock,
    mock_pyaudio_context_tts: MagicMock,
    mock_pyaudio_device_info: list[dict],
    mock_console: Console,
) -> None:
    """Test end-to-end speech synthesis with simplified mocks."""
    # Setup mock PyAudio
    mock_pyaudio_instance = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_context_speak.return_value.__enter__.return_value = mock_pyaudio_instance
    mock_pyaudio_context_tts.return_value.__enter__.return_value = mock_pyaudio_instance

    # Setup mock Wyoming client
    mock_tts_client = MockTTSClient(b"fake audio data")
    mock_async_client_class.from_uri.return_value.__aenter__.return_value = mock_tts_client

    general_cfg = GeneralConfig(
        log_level="INFO",
        log_file=None,
        quiet=False,
    )
    general_cfg.__dict__["console"] = mock_console
    tts_config = TTSConfig(
        enabled=True,
        server_ip="mock-host",
        server_port=10200,
        voice_name=None,
        language=None,
        speaker=None,
        output_device_index=None,
        output_device_name=None,
        speed=1.0,
    )
    file_config = FileConfig(save_file=None)

    await async_main(
        general_cfg=general_cfg,
        text="Hello, world!",
        tts_config=tts_config,
        file_config=file_config,
        list_devices=False,
    )

    # Verify that the audio was "played"
    mock_async_client_class.from_uri.assert_called_once_with("tcp://mock-host:10200")
    assert mock_pyaudio_instance.streams[0].get_written_data()
