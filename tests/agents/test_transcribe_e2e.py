"""End-to-end tests for the transcribe agent with minimal mocking."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_cli.agents.transcribe import async_main
from tests.mocks.audio import MockPyAudio
from tests.mocks.wyoming import MockASRClient


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
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

    with (
        patch("asyncio.Event.wait", new_callable=AsyncMock),
        patch(
            "agent_cli.asr.send_audio",
            new_callable=AsyncMock,
        ) as mock_send_audio,
        patch(
            "agent_cli.asr.receive_text",
            new_callable=AsyncMock,
        ) as mock_receive_text,
    ):
        mock_receive_text.return_value = "This is a test transcription."
        await async_main(
            device_index=0,
            asr_server_ip="mock-host",
            asr_server_port=10300,
            clipboard=False,
            quiet=False,
            llm=False,
            model="",
            ollama_host="",
            console=None,
            p=mock_pyaudio_class.return_value,
        )
    # A more robust test would be to capture the output and assert on it,
    # but for now, we'll just assert that the mock was called.
    mock_async_client_class.from_uri.assert_called_once()
    mock_send_audio.assert_awaited_once()
    mock_receive_text.assert_awaited_once()
