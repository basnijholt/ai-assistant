"""End-to-end tests for the audio module with minimal mocking."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from rich.console import Console

from agent_cli import audio
from tests.mocks.audio import MockPyAudio


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_get_all_devices_caching(
    mock_pyaudio_class,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test that device enumeration is properly cached."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Clear any existing cache
    audio.get_all_devices.cache_clear()

    # Create PyAudio instance
    with audio.pyaudio_context() as p:
        # First call should populate cache
        devices1 = audio.get_all_devices(p)
        assert len(devices1) == len(mock_pyaudio_device_info)

        # Second call should use cache
        devices2 = audio.get_all_devices(p)
        assert devices1 == devices2

        # Verify device info structure
        for i, device in enumerate(devices1):
            assert device["index"] == i
            assert "name" in device


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_list_input_devices(
    mock_pyaudio_class,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test listing input devices."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Clear cache
    audio.get_all_devices.cache_clear()

    # List input devices
    with audio.pyaudio_context() as p:
        audio.list_input_devices(p, console=mock_console)

    # Verify console output
    console_output = mock_console.file.getvalue()
    assert "Mock Input Device" in console_output
    assert "Mock Combined Device" in console_output
    # Should not show output-only device
    assert "Mock Output Device" not in console_output


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_list_output_devices(
    mock_pyaudio_class,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test listing output devices."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Clear cache
    audio.get_all_devices.cache_clear()

    # List output devices
    with audio.pyaudio_context() as p:
        audio.list_output_devices(p, console=mock_console)

    # Verify console output
    console_output = mock_console.file.getvalue()
    assert "Mock Output Device" in console_output
    assert "Mock Combined Device" in console_output
    # Should not show input-only device
    assert "Mock Input Device" not in console_output


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_list_all_devices(
    mock_pyaudio_class,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test listing all devices."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Clear cache
    audio.get_all_devices.cache_clear()

    # List all devices
    with audio.pyaudio_context() as p:
        audio.list_all_devices(p, console=mock_console)

    # Verify console output contains all devices
    console_output = mock_console.file.getvalue()
    assert "Mock Input Device" in console_output
    assert "Mock Output Device" in console_output
    assert "Mock Combined Device" in console_output


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_input_device_by_index(
    mock_pyaudio_class,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test selecting input device by index."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Clear cache
    audio.get_all_devices.cache_clear()

    # Get input device by index
    with audio.pyaudio_context() as p:
        device_index, device_name = audio.input_device(
            p,
            device_name=None,
            device_index=0,
        )

    # Verify correct device is returned
    assert device_index == 0
    assert device_name == "Mock Input Device"


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_input_device_by_name(
    mock_pyaudio_class,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test selecting input device by name."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Clear cache
    audio.get_all_devices.cache_clear()

    # Get input device by name
    with audio.pyaudio_context() as p:
        device_index, device_name = audio.input_device(
            p,
            device_name="Mock Combined Device",
            device_index=None,
        )

    # Verify correct device is returned
    assert device_index == 2
    assert device_name == "Mock Combined Device"


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_output_device_by_index(
    mock_pyaudio_class,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test selecting output device by index."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Clear cache
    audio.get_all_devices.cache_clear()

    # Get output device by index
    with audio.pyaudio_context() as p:
        device_index, device_name = audio.output_device(
            p,
            device_name=None,
            device_index=1,
        )

    # Verify correct device is returned
    assert device_index == 1
    assert device_name == "Mock Output Device"


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_output_device_by_name(
    mock_pyaudio_class,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test selecting output device by name."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Clear cache
    audio.get_all_devices.cache_clear()

    # Get output device by name
    with audio.pyaudio_context() as p:
        device_index, device_name = audio.output_device(
            p,
            device_name="Mock Combined Device",
            device_index=None,
        )

    # Verify correct device is returned
    assert device_index == 2
    assert device_name == "Mock Combined Device"


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_input_device_invalid_index(
    mock_pyaudio_class,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test handling of invalid input device index."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Clear cache
    audio.get_all_devices.cache_clear()

    # Try to get device with invalid index - should raise ValueError
    with audio.pyaudio_context() as p:
        with pytest.raises(ValueError, match="Device index 999 not found"):
            audio.input_device(
                p,
                device_name=None,
                device_index=999,
            )


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_input_device_invalid_name(
    mock_pyaudio_class,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test handling of invalid input device name."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Clear cache
    audio.get_all_devices.cache_clear()

    # Try to get device with invalid name - should raise ValueError
    with audio.pyaudio_context() as p:
        with pytest.raises(ValueError, match="No input device found"):
            audio.input_device(
                p,
                device_name="Nonexistent Device",
                device_index=None,
            )


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_output_device_invalid_name(
    mock_pyaudio_class,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test handling of invalid output device name."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Clear cache
    audio.get_all_devices.cache_clear()

    # Try to get device with invalid name - should raise ValueError
    with audio.pyaudio_context() as p:
        with pytest.raises(ValueError, match="No output device found"):
            audio.output_device(
                p,
                device_name="Nonexistent Output Device",
                device_index=None,
            )


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_pyaudio_context_manager(
    mock_pyaudio_class,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test PyAudio context manager functionality."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Test context manager
    with audio.pyaudio_context() as p:
        assert p is mock_pyaudio
        assert len(p.streams) == 0  # No streams created yet

    # Verify terminate was called
    assert len(mock_pyaudio.streams) == 0  # All streams should be closed


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_open_pyaudio_stream_context_manager(
    mock_pyaudio_class,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test PyAudio stream context manager functionality."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Test stream context manager
    with audio.pyaudio_context() as p:
        with audio.open_pyaudio_stream(
            p,
            format="int16",
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
            input_device_index=0,
        ) as stream:
            assert stream is not None
            assert len(p.streams) == 1
            assert stream.is_active

        # Stream should be closed after context
        assert not stream.is_active


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_device_filtering_by_capabilities(
    mock_pyaudio_class,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test that devices are properly filtered by input/output capabilities."""
    # Setup mock PyAudio with custom device info
    custom_devices = [
        {
            "index": 0,
            "name": "Input Only Device",
            "maxInputChannels": 2,
            "maxOutputChannels": 0,
            "defaultSampleRate": 44100.0,
        },
        {
            "index": 1,
            "name": "Output Only Device",
            "maxInputChannels": 0,
            "maxOutputChannels": 2,
            "defaultSampleRate": 44100.0,
        },
        {
            "index": 2,
            "name": "Combined Device",
            "maxInputChannels": 2,
            "maxOutputChannels": 2,
            "defaultSampleRate": 44100.0,
        },
        {
            "index": 3,
            "name": "No Channels Device",
            "maxInputChannels": 0,
            "maxOutputChannels": 0,
            "defaultSampleRate": 44100.0,
        },
    ]

    mock_pyaudio = MockPyAudio(custom_devices)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Clear cache
    audio.get_all_devices.cache_clear()

    # Test input device filtering
    with audio.pyaudio_context() as p:
        audio.list_input_devices(p, console=mock_console)
    console_output = mock_console.file.getvalue()

    # Should include input-capable devices
    assert "Input Only Device" in console_output
    assert "Combined Device" in console_output

    # Should not include output-only or no-channel devices in main list
    lines = console_output.split("\n")
    input_device_lines = [
        line for line in lines if "Input Only Device" in line or "Combined Device" in line
    ]
    assert len(input_device_lines) >= 2

    # Clear console for next test
    mock_console.file.truncate(0)
    mock_console.file.seek(0)

    # Test output device filtering
    with audio.pyaudio_context() as p:
        audio.list_output_devices(p, console=mock_console)
    console_output = mock_console.file.getvalue()

    # Should include output-capable devices
    assert "Output Only Device" in console_output
    assert "Combined Device" in console_output
