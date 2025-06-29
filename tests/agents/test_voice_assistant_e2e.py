"""End-to-end tests for the voice assistant agent with simplified mocks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_cli.agents.voice_assistant import async_main
from tests.mocks.audio import MockPyAudio
from tests.mocks.llm import MockLLMAgent
from tests.mocks.wyoming import MockASRClient, MockTTSClient


@pytest.mark.asyncio
@patch("agent_cli.tts.AsyncClient")
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.llm.build_agent")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_voice_assistant_e2e(
    mock_pyaudio_class: MagicMock,
    mock_build_agent: MagicMock,
    mock_asr_client_class: MagicMock,
    mock_tts_client_class: MagicMock,
    mock_pyaudio_device_info: list[dict],
    llm_responses: dict[str, str],
) -> None:
    """Test end-to-end voice assistant functionality with simplified mocks."""
    # Setup mock PyAudio
    mock_pyaudio_class.return_value = MockPyAudio(mock_pyaudio_device_info)

    # Setup mock LLM agent
    mock_build_agent.return_value = MockLLMAgent(llm_responses)

    # Setup mock Wyoming clients
    mock_asr_client = MockASRClient("this is a test")
    mock_tts_client = MockTTSClient(b"fake audio data")

    def from_uri_side_effect(uri: str) -> MockASRClient | MockTTSClient:
        if "asr" in uri:
            return mock_asr_client
        return mock_tts_client

    mock_asr_client_class.from_uri.side_effect = from_uri_side_effect
    mock_tts_client_class.from_uri.side_effect = from_uri_side_effect

    with (
        patch(
            "agent_cli.agents.voice_assistant.get_clipboard_text",
            return_value="test clipboard text",
        ),
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
        mock_receive_text.return_value = "this is a test"
        await async_main(
            console=None,
            device_index=0,
            device_name=None,
            list_devices=False,
            asr_server_ip="mock-host",
            asr_server_port=10300,
            model="test-model",
            ollama_host="http://localhost:11434",
            clipboard=True,
            enable_tts=True,
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
    mock_build_agent.assert_called_once()
    mock_asr_client_class.from_uri.assert_called()
    mock_tts_client_class.from_uri.assert_called()
    mock_send_audio.assert_awaited_once()
    mock_receive_text.assert_awaited_once()
