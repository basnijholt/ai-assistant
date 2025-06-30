"""Tests for the transcribe agent."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest

from agent_cli.agents import transcribe
from agent_cli.agents._config import ASRConfig, GeneralConfig, LLMConfig
from tests.mocks.wyoming import MockASRClient


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.agents.transcribe.pyperclip")
@patch("agent_cli.agents.transcribe.pyaudio_context")
@patch("agent_cli.agents.transcribe.input_device")
@patch("agent_cli.agents.transcribe.signal_handling_context")
async def test_transcribe_main(
    mock_signal_handling_context: MagicMock,
    mock_input_device: MagicMock,
    mock_pyaudio_context: MagicMock,
    mock_pyperclip: MagicMock,
    mock_async_client_class: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the main function of the transcribe agent."""
    # Mock the pyaudio context manager
    mock_pyaudio_instance = MagicMock()
    mock_pyaudio_context.return_value.__enter__.return_value = mock_pyaudio_instance

    # Mock the Wyoming client
    mock_asr_client = MockASRClient("hello world")
    mock_async_client_class.from_uri.return_value = mock_asr_client
    mock_input_device.return_value = (None, None)

    # Setup stop event
    stop_event = asyncio.Event()
    mock_signal_handling_context.return_value.__enter__.return_value = stop_event
    asyncio.get_event_loop().call_later(0.1, stop_event.set)

    # The function we are testing
    with caplog.at_level(logging.INFO):
        asr_config = ASRConfig(
            server_ip="localhost",
            server_port=12345,
            device_index=None,
            device_name=None,
            list_devices=False,
        )
        general_config = GeneralConfig(
            log_level="INFO",
            log_file=None,
            quiet=True,
            console=None,
            clipboard=True,
        )
        llm_config = LLMConfig(model="", ollama_host="")
        await transcribe.async_main(
            asr_config=asr_config,
            general_config=general_config,
            llm_config=llm_config,
            llm_enabled=False,
            p=mock_pyaudio_instance,
        )

    # Assertions
    assert "Copied transcript to clipboard." in caplog.text
    mock_pyperclip.copy.assert_called_once_with("hello world")
    mock_async_client_class.from_uri.assert_called_once_with("tcp://localhost:12345")
