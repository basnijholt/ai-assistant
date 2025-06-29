"""End-to-end tests for the speak agent with minimal mocking."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from agent_cli.agents import speak
from tests.mocks.audio import MockPyAudio
from tests.mocks.wyoming import MockWyomingAsyncClient, create_mock_audio_data


@pytest.mark.asyncio
@patch("wyoming.client.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_basic_functionality(
    mock_pyaudio_class,
    mock_async_client_class,
    mock_console: Console,
    mock_logger: logging.Logger,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test basic text-to-speech functionality end-to-end."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Configure TTS responses
    tts_responses = {
        "hello": synthetic_audio_data,
        "test": create_mock_audio_data("test message"),
    }

    # Configure TTS responses
    mock_client = MockWyomingAsyncClient.from_uri(
        "tcp://localhost:10200",
        tts_responses=tts_responses,
    )

    mock_async_client_class.from_uri.return_value = mock_client

    # Run TTS with timeout
    await asyncio.wait_for(
        speak.async_main(
            text="Hello, this is a test!",
            tts_server_ip="localhost",
            tts_server_port=10200,
            voice_name=None,
            tts_language=None,
            speaker=None,
            output_device_index=None,
            output_device_name=None,
            list_output_devices_flag=False,
            save_file=None,
            quiet=False,
            console=mock_console,
        ),
        timeout=5.0,
    )

    # Verify audio was "played" (written to stream)
    assert len(mock_pyaudio.streams) > 0
    output_stream = mock_pyaudio.streams[-1]  # Last created stream
    written_data = output_stream.get_written_data()
    assert len(written_data) > 0

    # Verify console output
    console_output = mock_console.file.getvalue()
    assert "synthesizing" in console_output.lower() or "playing" in console_output.lower()


@pytest.mark.asyncio
@patch("agent_cli.tts.AsyncClient", MockWyomingAsyncClient)
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_with_voice_parameters(
    mock_pyaudio_class,
    mock_console: Console,
    mock_logger: logging.Logger,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test TTS with specific voice parameters."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Configure TTS responses
    original_from_uri = MockWyomingAsyncClient.from_uri

    def mock_from_uri(uri, **kwargs):
        return original_from_uri(
            uri,
            tts_responses={"test": synthetic_audio_data},
            **kwargs,
        )

    MockWyomingAsyncClient.from_uri = mock_from_uri

    # Run TTS with voice parameters and timeout
    await asyncio.wait_for(
        speak.async_main(
            text="Test message with voice parameters",
            tts_server_ip="localhost",
            tts_server_port=10200,
            voice_name="female_voice",
            tts_language="en-US",
            speaker="alice",
            output_device_index=1,  # Specific output device
            output_device_name=None,
            list_output_devices_flag=False,
            save_file=None,
            quiet=False,
            console=mock_console,
        ),
        timeout=5.0,
    )

    # Verify synthesis completed
    assert len(mock_pyaudio.streams) > 0
    output_stream = mock_pyaudio.streams[-1]
    written_data = output_stream.get_written_data()
    assert len(written_data) > 0


@pytest.mark.asyncio
@patch("agent_cli.tts.AsyncClient", MockWyomingAsyncClient)
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_save_to_file(
    mock_pyaudio_class,
    mock_console: Console,
    mock_logger: logging.Logger,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test saving TTS output to file."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Configure TTS responses
    original_from_uri = MockWyomingAsyncClient.from_uri

    def mock_from_uri(uri, **kwargs):
        return original_from_uri(
            uri,
            tts_responses={"save": synthetic_audio_data},
            **kwargs,
        )

    MockWyomingAsyncClient.from_uri = mock_from_uri

    # Create temporary file for output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        # Run TTS with file saving and timeout
        await asyncio.wait_for(
            speak.async_main(
                text="This message will be saved to file",
                tts_server_ip="localhost",
                tts_server_port=10200,
                voice_name=None,
                tts_language=None,
                speaker=None,
                output_device_index=None,
                output_device_name=None,
                list_output_devices_flag=False,
                save_file=str(temp_path),
                quiet=False,
                console=mock_console,
            ),
            timeout=5.0,
        )

        # Verify file was created and contains data
        assert temp_path.exists()
        assert temp_path.stat().st_size > 0

        # Verify console shows file save message
        console_output = mock_console.file.getvalue()
        assert "saved" in console_output.lower() or str(temp_path) in console_output

    finally:
        # Clean up
        if temp_path.exists():
            temp_path.unlink()


@pytest.mark.asyncio
@patch("agent_cli.tts.AsyncClient", MockWyomingAsyncClient)
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_device_listing(
    mock_pyaudio_class,
    mock_console: Console,
    mock_logger: logging.Logger,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test audio device listing functionality."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Run device listing (should not attempt TTS) with timeout
    await asyncio.wait_for(
        speak.async_main(
            text="This should not be synthesized",
            tts_server_ip="localhost",
            tts_server_port=10200,
            voice_name=None,
            tts_language=None,
            speaker=None,
            output_device_index=None,
            output_device_name=None,
            list_output_devices_flag=True,  # List devices instead of speaking
            save_file=None,
            quiet=False,
            console=mock_console,
        ),
        timeout=3.0,
    )

    # Verify device list was displayed
    console_output = mock_console.file.getvalue()
    assert "Mock Output Device" in console_output
    assert "Mock Combined Device" in console_output

    # Should not have created audio streams for TTS
    assert len(mock_pyaudio.streams) == 0


@pytest.mark.asyncio
@patch("agent_cli.tts.AsyncClient", MockWyomingAsyncClient)
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_device_name_selection(
    mock_pyaudio_class,
    mock_console: Console,
    mock_logger: logging.Logger,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test output device selection by name."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Configure TTS responses
    original_from_uri = MockWyomingAsyncClient.from_uri

    def mock_from_uri(uri, **kwargs):
        return original_from_uri(
            uri,
            tts_responses={"device": synthetic_audio_data},
            **kwargs,
        )

    MockWyomingAsyncClient.from_uri = mock_from_uri

    # Run TTS with device name selection
    await asyncio.wait_for(
        speak.async_main(
            text="Test device name selection",
            tts_server_ip="localhost",
            tts_server_port=10200,
            voice_name=None,
            tts_language=None,
            speaker=None,
            output_device_index=None,
            output_device_name="Mock Combined Device",  # Select by name
            list_output_devices_flag=False,
            save_file=None,
            quiet=False,
            console=mock_console,
        ),
        timeout=5.0,
    )

    # Verify synthesis completed
    assert len(mock_pyaudio.streams) > 0
    output_stream = mock_pyaudio.streams[-1]
    written_data = output_stream.get_written_data()
    assert len(written_data) > 0


@pytest.mark.asyncio
@patch("agent_cli.tts.AsyncClient", MockWyomingAsyncClient)
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_connection_error_handling(
    mock_pyaudio_class,
    mock_console: Console,
    mock_logger: logging.Logger,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test handling of connection errors to TTS server."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Mock connection refused error
    async def mock_from_uri_error(uri, **kwargs):
        raise ConnectionRefusedError("Connection refused")

    MockWyomingAsyncClient.from_uri = mock_from_uri_error

    # Run TTS - should handle error gracefully
    await asyncio.wait_for(
        speak.async_main(
            text="This should fail gracefully",
            tts_server_ip="localhost",
            tts_server_port=10200,
            voice_name=None,
            tts_language=None,
            speaker=None,
            output_device_index=None,
            output_device_name=None,
            list_output_devices_flag=False,
            save_file=None,
            quiet=False,
            console=mock_console,
        ),
        timeout=3.0,
    )

    # Should not have created audio streams on error
    assert len(mock_pyaudio.streams) == 0

    # Console should show error message
    console_output = mock_console.file.getvalue()
    assert "connection" in console_output.lower() or "error" in console_output.lower()


@pytest.mark.asyncio
@patch("agent_cli.tts.AsyncClient", MockWyomingAsyncClient)
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_quiet_mode(
    mock_pyaudio_class,
    mock_console: Console,
    mock_logger: logging.Logger,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test TTS in quiet mode."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Configure TTS responses
    original_from_uri = MockWyomingAsyncClient.from_uri

    def mock_from_uri(uri, **kwargs):
        return original_from_uri(
            uri,
            tts_responses={"quiet": synthetic_audio_data},
            **kwargs,
        )

    MockWyomingAsyncClient.from_uri = mock_from_uri

    # Run TTS in quiet mode
    await asyncio.wait_for(
        speak.async_main(
            text="Quiet mode test",
            tts_server_ip="localhost",
            tts_server_port=10200,
            voice_name=None,
            tts_language=None,
            speaker=None,
            output_device_index=None,
            output_device_name=None,
            list_output_devices_flag=False,
            save_file=None,
            quiet=True,  # Quiet mode
            console=None,  # No console in quiet mode
        ),
        timeout=5.0,
    )

    # Verify synthesis still completed
    assert len(mock_pyaudio.streams) > 0
    output_stream = mock_pyaudio.streams[-1]
    written_data = output_stream.get_written_data()
    assert len(written_data) > 0


@pytest.mark.asyncio
@patch("agent_cli.tts.AsyncClient", MockWyomingAsyncClient)
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_long_text(
    mock_pyaudio_class,
    mock_console: Console,
    mock_logger: logging.Logger,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test TTS with longer text content."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Create longer audio data for longer text
    long_audio_data = synthetic_audio_data * 3  # Simulate longer audio

    # Configure TTS responses
    original_from_uri = MockWyomingAsyncClient.from_uri

    def mock_from_uri(uri, **kwargs):
        return original_from_uri(
            uri,
            tts_responses={"long": long_audio_data},
            simulate_delay=0.05,  # Slightly longer delay for longer text
            **kwargs,
        )

    MockWyomingAsyncClient.from_uri = mock_from_uri

    # Long text content
    long_text = (
        "This is a much longer text that will be converted to speech. "
        "It contains multiple sentences and should result in a longer audio file. "
        "The text-to-speech system should handle this gracefully and produce "
        "appropriate audio output for the entire content."
    )

    # Run TTS with long text
    await asyncio.wait_for(
        speak.async_main(
            text=long_text,
            tts_server_ip="localhost",
            tts_server_port=10200,
            voice_name=None,
            tts_language=None,
            speaker=None,
            output_device_index=None,
            output_device_name=None,
            list_output_devices_flag=False,
            save_file=None,
            quiet=False,
            console=mock_console,
        ),
        timeout=5.0,
    )

    # Verify synthesis completed with longer audio
    assert len(mock_pyaudio.streams) > 0
    output_stream = mock_pyaudio.streams[-1]
    written_data = output_stream.get_written_data()
    assert len(written_data) > len(synthetic_audio_data)  # Should be longer
