"""End-to-end tests for the voice assistant agent with simplified mocks."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from agent_cli.agents._config import (
    ASRConfig,
    FileConfig,
    GeneralConfig,
    LLMConfig,
    TTSConfig,
)
from agent_cli.agents.voice_assistant import async_main
from tests.mocks.audio import MockPyAudio
from tests.mocks.llm import MockLLMAgent
from tests.mocks.wyoming import MockASRClient, MockTTSClient

if TYPE_CHECKING:
    from rich.console import Console


@pytest.mark.asyncio
@patch("agent_cli.tts.pyaudio_context")
@patch("agent_cli.llm.pyperclip.copy")
@patch("agent_cli.agents.voice_assistant.pyperclip")
@patch("agent_cli.agents.voice_assistant.pyaudio_context")
@patch("agent_cli.agents.voice_assistant.signal_handling_context")
@patch("agent_cli.tts.AsyncClient")
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.llm.build_agent")
async def test_voice_assistant_e2e(
    mock_build_agent: MagicMock,
    mock_asr_client_class: MagicMock,
    mock_tts_client_class: MagicMock,
    mock_signal_handling_context: MagicMock,
    mock_pyaudio_context_asr: MagicMock,
    mock_pyperclip: MagicMock,
    mock_llm_pyperclip_copy: MagicMock,
    mock_pyaudio_context_tts: MagicMock,
    mock_pyaudio_device_info: list[dict],
    llm_responses: dict[str, str],
    mock_console: Console,
) -> None:
    """Test end-to-end voice assistant functionality with simplified mocks."""
    # Setup mock PyAudio
    mock_pyaudio_instance = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_context_asr.return_value.__enter__.return_value = mock_pyaudio_instance
    mock_pyaudio_context_tts.return_value.__enter__.return_value = mock_pyaudio_instance

    # Setup mock LLM agent
    mock_llm_agent = MockLLMAgent(llm_responses)
    mock_build_agent.return_value = mock_llm_agent

    # Setup mock Wyoming clients
    mock_asr_client = MockASRClient("this is a test")
    mock_tts_client = MockTTSClient(b"fake audio data")
    mock_asr_client_class.from_uri.return_value = mock_asr_client
    mock_tts_client_class.from_uri.return_value.__aenter__.return_value = mock_tts_client

    # Setup stop event
    stop_event = asyncio.Event()
    mock_signal_handling_context.return_value.__enter__.return_value = stop_event
    asyncio.get_event_loop().call_later(0.1, stop_event.set)
    mock_pyperclip.paste.return_value = "this is the llm response"

    with patch(
        "agent_cli.agents.voice_assistant.get_clipboard_text",
        return_value="test clipboard text",
    ):
        general_cfg = GeneralConfig(
            log_level="INFO",
            log_file=None,
            quiet=False,
            clipboard=True,
        )
        general_cfg.__dict__["console"] = mock_console
        asr_config = ASRConfig(
            server_ip="mock-asr-host",
            server_port=10300,
            device_index=0,
            device_name=None,
            list_devices=False,
        )
        llm_config = LLMConfig(model="test-model", ollama_host="http://localhost:11434")
        tts_config = TTSConfig(
            enabled=True,
            server_ip="mock-tts-host",
            server_port=10200,
            voice_name=None,
            language=None,
            speaker=None,
            output_device_index=None,
            output_device_name=None,
            list_output_devices=False,
            speed=1.0,
        )
        file_config = FileConfig(save_file=None)

        await async_main(
            general_cfg=general_cfg,
            asr_config=asr_config,
            llm_config=llm_config,
            tts_config=tts_config,
            file_config=file_config,
        )

    # Assertions
    mock_build_agent.assert_called_once()
    mock_asr_client_class.from_uri.assert_called_once_with("tcp://mock-asr-host:10300")
    mock_tts_client_class.from_uri.assert_called_once_with("tcp://mock-tts-host:10200")
    assert mock_llm_agent.call_history
    assert mock_pyaudio_instance.streams[1].get_written_data()
    mock_pyperclip.paste.assert_called_once()
    mock_llm_pyperclip_copy.assert_called_once()
