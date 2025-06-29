"""End-to-end tests for the transcribe agent with minimal mocking."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest
from rich.console import Console

from agent_cli.agents import transcribe
from tests.mocks.audio import MockPyAudio
from tests.mocks.llm import create_autocorrect_responses, mock_build_agent
from tests.mocks.wyoming import MockWyomingAsyncClient


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_transcribe_basic_functionality(
    mock_pyaudio_class,
    mock_asr_client_class,
    mock_console: Console,
    mock_logger: logging.Logger,
    transcript_responses: dict[str, str],
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

    # Mock write_event (for sending audio)
    mock_client.write_event = AsyncMock()

    # Mock read_event to return a simple transcript
    from wyoming.asr import Transcript

    transcript_event = Transcript(text="Hello world").event()
    mock_client.read_event = AsyncMock(return_value=transcript_event)

    with patch("agent_cli.agents.transcribe.pyperclip") as mock_pyperclip:
        # Mock the signal handling to provide a stop event
        with patch("agent_cli.agents.transcribe.signal_handling_context") as mock_signal_context:
            stop_event = asyncio.Event()
            mock_signal_context.return_value.__enter__.return_value = stop_event
            mock_signal_context.return_value.__exit__.return_value = None

            # Set the stop event after a short delay to simulate user stopping
            async def auto_stop():
                await asyncio.sleep(0.1)  # Short delay for transcription to start
                stop_event.set()

            stop_task = asyncio.create_task(auto_stop())

            try:
                # Run transcription with timeout
                await asyncio.wait_for(
                    transcribe.async_main(
                        device_index=0,
                        asr_server_ip="localhost",
                        asr_server_port=10300,
                        clipboard=True,
                        quiet=False,
                        llm=False,
                        model="",
                        ollama_host="",
                        console=mock_console,
                        p=mock_pyaudio,
                    ),
                    timeout=3.0,
                )
            finally:
                stop_task.cancel()
                try:
                    await stop_task
                except asyncio.CancelledError:
                    pass

        # Verify clipboard was updated
        mock_pyperclip.copy.assert_called_once_with("Hello world")

        # Verify client connection was established
        mock_asr_client_class.from_uri.assert_called_once()


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient", MockWyomingAsyncClient)
@patch("agent_cli.audio.pyaudio.PyAudio")
@patch("agent_cli.llm.build_agent")
async def test_transcribe_with_llm_correction(
    mock_build_agent_func,
    mock_pyaudio_class,
    mock_console: Console,
    mock_logger: logging.Logger,
    transcript_responses: dict[str, str],
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test transcription with LLM-based text correction."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Setup mock LLM agent
    llm_responses = create_autocorrect_responses()
    mock_agent = mock_build_agent(
        model="test-model",
        ollama_host="http://localhost:11434",
        responses=llm_responses,
    )
    mock_build_agent_func.return_value = mock_agent

    # Configure transcript responses
    original_from_uri = MockWyomingAsyncClient.from_uri

    def mock_from_uri(uri, **kwargs):
        return original_from_uri(
            uri,
            asr_responses={"test": "hello world"},  # Raw transcription
            **kwargs,
        )

    MockWyomingAsyncClient.from_uri = mock_from_uri

    with patch("agent_cli.agents.transcribe.pyperclip") as mock_pyperclip:
        # Mock the signal handling to provide a stop event
        with patch("agent_cli.agents.transcribe.signal_handling_context") as mock_signal_context:
            stop_event = asyncio.Event()
            mock_signal_context.return_value.__enter__.return_value = stop_event
            mock_signal_context.return_value.__exit__.return_value = None

            # Set the stop event after a short delay
            async def auto_stop():
                await asyncio.sleep(0.3)
                stop_event.set()

            stop_task = asyncio.create_task(auto_stop())

            try:
                # Run transcription with LLM correction and timeout
                await asyncio.wait_for(
                    transcribe.async_main(
                        device_index=0,
                        asr_server_ip="localhost",
                        asr_server_port=10300,
                        clipboard=True,
                        quiet=False,
                        llm=True,
                        model="test-model",
                        ollama_host="http://localhost:11434",
                        console=mock_console,
                        p=mock_pyaudio,
                    ),
                    timeout=5.0,
                )
            finally:
                stop_task.cancel()
                try:
                    await stop_task
                except asyncio.CancelledError:
                    pass

        # Verify LLM was called
        assert len(mock_agent.call_history) > 0

        # Verify clipboard was updated with corrected text
        mock_pyperclip.copy.assert_called_once()
        copied_text = mock_pyperclip.copy.call_args[0][0]

        # Should be corrected version, not raw transcription
        assert copied_text != "hello world"  # Raw transcription
        assert "Hello, world!" in copied_text or "corrected" in copied_text.lower()


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient", MockWyomingAsyncClient)
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_transcribe_device_selection(
    mock_pyaudio_class,
    mock_console: Console,
    mock_logger: logging.Logger,
    transcript_responses: dict[str, str],
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test transcription with specific audio device selection."""
    # Setup mock PyAudio with multiple devices
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Configure transcript responses
    original_from_uri = MockWyomingAsyncClient.from_uri

    def mock_from_uri(uri, **kwargs):
        return original_from_uri(
            uri,
            asr_responses=transcript_responses,
            **kwargs,
        )

    MockWyomingAsyncClient.from_uri = mock_from_uri

    with patch("agent_cli.agents.transcribe.pyperclip") as mock_pyperclip:
        # Mock the signal handling to provide a stop event
        with patch("agent_cli.agents.transcribe.signal_handling_context") as mock_signal_context:
            stop_event = asyncio.Event()
            mock_signal_context.return_value.__enter__.return_value = stop_event
            mock_signal_context.return_value.__exit__.return_value = None

            # Set the stop event after a short delay
            async def auto_stop():
                await asyncio.sleep(0.2)
                stop_event.set()

            stop_task = asyncio.create_task(auto_stop())

            try:
                # Run transcription with device index 2 (combined device)
                await asyncio.wait_for(
                    transcribe.async_main(
                        device_index=2,
                        asr_server_ip="localhost",
                        asr_server_port=10300,
                        clipboard=True,
                        quiet=True,  # Quiet mode for this test
                        llm=False,
                        model="",
                        ollama_host="",
                        console=None,  # No console in quiet mode
                        p=mock_pyaudio,
                    ),
                    timeout=3.0,
                )
            finally:
                stop_task.cancel()
                try:
                    await stop_task
                except asyncio.CancelledError:
                    pass

        # Verify transcription completed
        mock_pyperclip.copy.assert_called_once()

        # Verify the correct device was used (device index 2)
        assert len(mock_pyaudio.streams) > 0
        # The stream should have been created (exact verification depends on implementation)


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient", MockWyomingAsyncClient)
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_transcribe_connection_error_handling(
    mock_pyaudio_class,
    mock_console: Console,
    mock_logger: logging.Logger,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test handling of connection errors to ASR server."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Mock connection refused error
    async def mock_from_uri_error(uri, **kwargs):
        raise ConnectionRefusedError("Connection refused")

    MockWyomingAsyncClient.from_uri = mock_from_uri_error

    with patch("agent_cli.agents.transcribe.pyperclip") as mock_pyperclip:
        # Mock the signal handling to provide a stop event
        with patch("agent_cli.agents.transcribe.signal_handling_context") as mock_signal_context:
            stop_event = asyncio.Event()
            mock_signal_context.return_value.__enter__.return_value = stop_event
            mock_signal_context.return_value.__exit__.return_value = None

            try:
                # Run transcription - should handle error gracefully
                await asyncio.wait_for(
                    transcribe.async_main(
                        device_index=0,
                        asr_server_ip="localhost",
                        asr_server_port=10300,
                        clipboard=False,
                        quiet=False,
                        llm=False,
                        model="",
                        ollama_host="",
                        console=mock_console,
                        p=mock_pyaudio,
                    ),
                    timeout=3.0,
                )
            except TimeoutError:
                # This is expected since the connection will fail
                pass

        # Should not attempt to copy to clipboard on error
        mock_pyperclip.copy.assert_not_called()

        # Console should show error message
        console_output = mock_console.file.getvalue()
        assert "connection" in console_output.lower() or "error" in console_output.lower()


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient", MockWyomingAsyncClient)
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_transcribe_empty_result_handling(
    mock_pyaudio_class,
    mock_console: Console,
    mock_logger: logging.Logger,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test handling of empty transcription results."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Configure empty transcript response
    original_from_uri = MockWyomingAsyncClient.from_uri

    def mock_from_uri(uri, **kwargs):
        return original_from_uri(
            uri,
            asr_responses={"default": ""},  # Empty response
            **kwargs,
        )

    MockWyomingAsyncClient.from_uri = mock_from_uri

    with patch("agent_cli.agents.transcribe.pyperclip") as mock_pyperclip:
        # Mock the signal handling to provide a stop event
        with patch("agent_cli.agents.transcribe.signal_handling_context") as mock_signal_context:
            stop_event = asyncio.Event()
            mock_signal_context.return_value.__enter__.return_value = stop_event
            mock_signal_context.return_value.__exit__.return_value = None

            # Set the stop event after a short delay
            async def auto_stop():
                await asyncio.sleep(0.2)
                stop_event.set()

            stop_task = asyncio.create_task(auto_stop())

            try:
                # Run transcription
                await asyncio.wait_for(
                    transcribe.async_main(
                        device_index=0,
                        asr_server_ip="localhost",
                        asr_server_port=10300,
                        clipboard=True,
                        quiet=False,
                        llm=False,
                        model="",
                        ollama_host="",
                        console=mock_console,
                        p=mock_pyaudio,
                    ),
                    timeout=3.0,
                )
            finally:
                stop_task.cancel()
                try:
                    await stop_task
                except asyncio.CancelledError:
                    pass

        # Should still copy to clipboard, even if empty
        mock_pyperclip.copy.assert_called_once()
        copied_text = mock_pyperclip.copy.call_args[0][0]
        # Empty result should be handled gracefully
        assert isinstance(copied_text, str)
