"""End-to-end tests for the speak agent with minimal mocking."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Never
from unittest.mock import Mock, patch

import pytest

from agent_cli.agents import speak
from tests.mocks.audio import MockPyAudio
from tests.mocks.wyoming import MockWyomingAsyncClient

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import Console


@pytest.mark.asyncio
@patch("wyoming.client.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_basic_functionality(
    mock_pyaudio_class: Mock,
    mock_async_client_class: Mock,
    mock_console: Console,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test basic text-to-speech functionality."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Configure TTS responses
    mock_client = MockWyomingAsyncClient.from_uri(
        "tcp://localhost:10200",
        tts_responses={"test": synthetic_audio_data},
    )

    mock_async_client_class.from_uri.return_value = mock_client

    # Test basic TTS
    with pytest.raises(asyncio.TimeoutError):
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
            timeout=3.0,
        )


@pytest.mark.asyncio
@patch("wyoming.client.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_save_to_file(
    mock_pyaudio_class: Mock,
    mock_async_client_class: Mock,
    mock_console: Console,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
    tmp_path: Path,
) -> None:
    """Test saving TTS output to a file."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Setup TTS responses
    temp_path = tmp_path / "test_output.wav"

    mock_client = MockWyomingAsyncClient.from_uri(
        "tcp://localhost:10200",
        tts_responses={"save": synthetic_audio_data},
    )

    mock_async_client_class.from_uri.return_value = mock_client

    # Test TTS with file saving
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            speak.async_main(
                text="Save this audio to a file",
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
            timeout=3.0,
        )

    # Verify file was created (the audio data would have been processed)
    # Note: The actual file creation is mocked, but we can verify the path was handled


@pytest.mark.asyncio
@patch("wyoming.client.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_list_devices(
    mock_pyaudio_class: Mock,
    mock_async_client_class: Mock,
    mock_console: Console,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test listing output devices instead of speaking."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    mock_client = MockWyomingAsyncClient.from_uri(
        "tcp://localhost:10200",
        tts_responses={"list": synthetic_audio_data},
    )

    mock_async_client_class.from_uri.return_value = mock_client

    # Test device listing (should not speak)
    await speak.async_main(
        text="This text should not be spoken",
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
    )

    # Should have listed devices instead of speaking
    # (No timeout needed since device listing is synchronous)


@pytest.mark.asyncio
@patch("wyoming.client.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_device_selection(
    mock_pyaudio_class: Mock,
    mock_async_client_class: Mock,
    mock_console: Console,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test TTS with specific output device selection."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    mock_client = MockWyomingAsyncClient.from_uri(
        "tcp://localhost:10200",
        tts_responses={"device": synthetic_audio_data},
    )

    mock_async_client_class.from_uri.return_value = mock_client

    # Test TTS with device selection
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            speak.async_main(
                text="Test with specific device",
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
            timeout=3.0,
        )


@pytest.mark.asyncio
@patch("wyoming.client.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_connection_error_handling(
    mock_pyaudio_class: Mock,
    mock_async_client_class: Mock,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test graceful handling of TTS server connection errors."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Mock connection refused error
    async def mock_from_uri_error(uri: str, **kwargs: dict) -> Never:  # noqa: ARG001
        msg = "Connection refused"
        raise ConnectionRefusedError(msg)

    mock_async_client_class.from_uri = mock_from_uri_error

    # Test TTS with connection error - should handle gracefully
    with pytest.raises(asyncio.TimeoutError):
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


@pytest.mark.asyncio
@patch("wyoming.client.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_quiet_mode(
    mock_pyaudio_class: Mock,
    mock_async_client_class: Mock,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test TTS in quiet mode (no console output)."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    mock_client = MockWyomingAsyncClient.from_uri(
        "tcp://localhost:10200",
        tts_responses={"quiet": synthetic_audio_data},
    )

    mock_async_client_class.from_uri.return_value = mock_client

    # Test quiet mode TTS
    with pytest.raises(asyncio.TimeoutError):
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
            timeout=3.0,
        )


@pytest.mark.asyncio
@patch("wyoming.client.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
async def test_speak_long_text(
    mock_pyaudio_class: Mock,
    mock_async_client_class: Mock,
    mock_console: Console,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test TTS with longer text content."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Create longer audio data for longer text
    long_text = (
        "This is a much longer piece of text that should take more time to synthesize and play back. "
        * 5
    )
    long_audio_data = synthetic_audio_data * 10  # Simulate longer audio

    mock_client = MockWyomingAsyncClient.from_uri(
        "tcp://localhost:10200",
        tts_responses={"long": long_audio_data},
    )

    mock_async_client_class.from_uri.return_value = mock_client

    # Test TTS with long text
    with pytest.raises(asyncio.TimeoutError):
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
            timeout=3.0,
        )
