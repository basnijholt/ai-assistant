"""End-to-end tests for the transcribe agent with minimal mocking."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Never
from unittest.mock import AsyncMock, Mock, patch

import pytest
from rich.console import Console
from wyoming.asr import Transcript

from agent_cli.agents import transcribe
from tests.mocks.audio import MockPyAudio
from tests.mocks.llm import create_autocorrect_responses, mock_build_agent
from tests.mocks.wyoming import MockWyomingAsyncClient

if TYPE_CHECKING:
    from rich.console import Console


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_transcribe_basic_functionality(
    mock_pyaudio_class: Mock,
    mock_asr_client_class: Mock,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test basic transcription functionality end-to-end."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Create a simple mock client that returns a transcript
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    # Mock the from_uri class method to return the mock client directly (not a coroutine)
    mock_asr_client_class.from_uri.return_value = mock_client

    # Mock read_event to return a simple transcript
    transcript_event = Transcript(text="Hello world").event()
    mock_client.read_event = AsyncMock(return_value=transcript_event)
    mock_client.write_event = AsyncMock()

    # Test basic transcription
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            transcribe.async_main(
                device_index=0,
                device_name=None,
                asr_server_ip="localhost",
                asr_server_port=10300,
                clipboard=False,
                live=False,
                quiet=False,
                llm=False,
                model="",
                ollama_host="",
                console=mock_console,
            ),
            timeout=3.0,
        )

    # Verify the client was used properly
    mock_client.write_event.assert_called()


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
@patch("agent_cli.llm.build_agent")
async def test_transcribe_with_llm_correction(
    mock_build_agent_func: Mock,
    mock_pyaudio_class: Mock,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test transcription with LLM correction enabled."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Setup mock LLM agent
    llm_responses = create_autocorrect_responses()
    mock_agent = mock_build_agent(
        "test-model",
        "http://localhost:11434",
        llm_responses,
    )
    mock_build_agent_func.return_value = mock_agent

    # Setup Wyoming client mock
    original_from_uri = MockWyomingAsyncClient.from_uri

    def mock_from_uri(uri: str, **kwargs: dict) -> MockWyomingAsyncClient:  # type: ignore[misc]
        return original_from_uri(
            uri,
            asr_responses={"default": "test transcription with typos"},
            **kwargs,
        )

    MockWyomingAsyncClient.from_uri = mock_from_uri

    # Test transcription with LLM correction
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            transcribe.async_main(
                device_index=0,
                device_name=None,
                asr_server_ip="localhost",
                asr_server_port=10300,
                clipboard=False,
                live=False,
                quiet=False,
                llm=True,  # Enable LLM correction
                model="test-model",
                ollama_host="http://localhost:11434",
                console=mock_console,
            ),
            timeout=3.0,
        )

    # Verify LLM agent was called
    assert mock_agent.call_count > 0


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_transcribe_device_selection(
    mock_pyaudio_class: Mock,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test device selection for transcription."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Setup Wyoming client mock
    original_from_uri = MockWyomingAsyncClient.from_uri

    def mock_from_uri(uri: str, **kwargs: dict) -> MockWyomingAsyncClient:  # type: ignore[misc]
        return original_from_uri(
            uri,
            asr_responses={"default": "device test transcription"},
            **kwargs,
        )

    MockWyomingAsyncClient.from_uri = mock_from_uri

    # Test transcription with specific device
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            transcribe.async_main(
                device_index=None,  # No index specified
                device_name="Mock Combined Device",  # Select by name
                asr_server_ip="localhost",
                asr_server_port=10300,
                clipboard=False,
                live=False,
                quiet=False,
                llm=False,
                model="",
                ollama_host="",
                console=mock_console,
            ),
            timeout=3.0,
        )


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_transcribe_connection_error_handling(
    mock_pyaudio_class: Mock,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test graceful handling of connection errors."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Mock connection refused error
    async def mock_from_uri_error(uri: str, **kwargs: dict) -> Never:  # noqa: ARG001
        msg = "Connection refused"
        raise ConnectionRefusedError(msg)

    MockWyomingAsyncClient.from_uri = mock_from_uri_error

    # Mock signal handling to prevent hanging
    with patch("agent_cli.agents.transcribe.signal_handling_context") as mock_signal_context:
        stop_event = asyncio.Event()
        stop_event.set()  # Stop immediately
        mock_signal_context.return_value.__enter__.return_value = stop_event
        mock_signal_context.return_value.__exit__.return_value = None

        with contextlib.suppress(TimeoutError):
            # Run transcription - should handle error gracefully
            await asyncio.wait_for(
                transcribe.async_main(
                    device_index=0,
                    device_name=None,
                    asr_server_ip="localhost",
                    asr_server_port=10300,
                    clipboard=False,
                    live=False,
                    quiet=False,
                    llm=False,
                    model="",
                    ollama_host="",
                    console=mock_console,
                ),
                timeout=3.0,
            )

        # Should not attempt to copy to clipboard on error


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_transcribe_empty_result_handling(
    mock_pyaudio_class: Mock,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test handling of empty transcription results."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Setup Wyoming client mock to return empty results
    original_from_uri = MockWyomingAsyncClient.from_uri

    def mock_from_uri(uri: str, **kwargs: dict) -> MockWyomingAsyncClient:  # type: ignore[misc]
        return original_from_uri(
            uri,
            asr_responses={"default": ""},  # Empty response
            **kwargs,
        )

    MockWyomingAsyncClient.from_uri = mock_from_uri

    # Test transcription with empty result
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            transcribe.async_main(
                device_index=0,
                device_name=None,
                asr_server_ip="localhost",
                asr_server_port=10300,
                clipboard=False,
                live=False,
                quiet=False,
                llm=False,
                model="",
                ollama_host="",
                console=mock_console,
            ),
            timeout=3.0,
        )
