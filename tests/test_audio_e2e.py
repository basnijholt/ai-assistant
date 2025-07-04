"""End-to-end tests for the audio module with minimal mocking."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from agent_cli import audio
from tests.mocks.audio import MockPyAudio

if TYPE_CHECKING:
    from rich.console import Console


@pytest.fixture
def mock_pyaudio_with_cache_clear() -> None:
    """Clear the audio device cache before each test."""
    audio.get_all_devices.cache_clear()


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_get_all_devices_caching(
    mock_pyaudio_class: Mock,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test that device enumeration is cached for performance."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    with audio.pyaudio_context() as p:
        # First call should hit PyAudio
        devices1 = audio.get_all_devices(p)

        # Second call should use cached results
        devices2 = audio.get_all_devices(p)

        # Results should be identical
        assert devices1 == devices2
        assert len(devices1) == len(mock_pyaudio_device_info)


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_list_input_devices(
    mock_pyaudio_class: Mock,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test listing input devices."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    with audio.pyaudio_context() as p:
        # Test listing input devices
        audio.list_input_devices(p, mock_console)

    # Verify console output contains device information
    # This is more of an integration test to ensure no exceptions are raised


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_list_output_devices(
    mock_pyaudio_class: Mock,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test listing output devices."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    with audio.pyaudio_context() as p:
        # Test listing output devices
        audio.list_output_devices(p, mock_console)

    # Verify no exceptions are raised


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_list_all_devices(
    mock_pyaudio_class: Mock,
    mock_console: Console,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test listing all audio devices."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    with audio.pyaudio_context() as p:
        # Test listing all devices
        audio.list_all_devices(p, mock_console)

    # Verify no exceptions are raised


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_input_device_by_index(
    mock_pyaudio_class: Mock,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test selecting input device by index."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    with audio.pyaudio_context() as p:
        # Test getting device by valid index
        input_device_index, input_device_name = audio.input_device(
            p,
            input_device_name=None,
            input_device_index=0,
        )

        # Should return the first input-capable device
        expected_device = next(
            dev for dev in mock_pyaudio_device_info if dev["maxInputChannels"] > 0
        )

        assert input_device_name == expected_device["name"]
        assert input_device_index == expected_device["index"]


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_input_device_by_name(
    mock_pyaudio_class: Mock,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test selecting input device by name."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    with audio.pyaudio_context() as p:
        # Find an input device to test with
        input_device = next(dev for dev in mock_pyaudio_device_info if dev["maxInputChannels"] > 0)

        # Test getting device by name
        input_device_index, input_device_name = audio.input_device(
            p,
            input_device_name=input_device["name"],
            input_device_index=None,
        )

        assert input_device_name == input_device["name"]
        assert input_device_index == input_device["index"]


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_output_device_by_index(
    mock_pyaudio_class: Mock,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test selecting output device by index."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    with audio.pyaudio_context() as p:
        # Test getting device by valid index
        input_device_index, input_device_name = audio.output_device(
            p,
            input_device_name=None,
            input_device_index=1,
        )

        # Should return the device at index 1 if it has output channels
        expected_device = next(
            dev
            for dev in mock_pyaudio_device_info
            if dev["index"] == 1 and dev["maxOutputChannels"] > 0
        )

        assert input_device_name == expected_device["name"]
        assert input_device_index == expected_device["index"]


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_output_device_by_name(
    mock_pyaudio_class: Mock,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test selecting output device by name."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    with audio.pyaudio_context() as p:
        # Find an output device to test with
        output_device = next(
            dev for dev in mock_pyaudio_device_info if dev["maxOutputChannels"] > 0
        )

        # Test getting device by name
        input_device_index, input_device_name = audio.output_device(
            p,
            input_device_name=output_device["name"],
            input_device_index=None,
        )

        assert input_device_name == output_device["name"]
        assert input_device_index == output_device["index"]


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_input_device_invalid_index(
    mock_pyaudio_class: Mock,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test error handling for invalid device index."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Try to get device with invalid index - should raise ValueError
    with (
        audio.pyaudio_context() as p,
        pytest.raises(ValueError, match="Device index 999 not found"),
    ):
        audio.input_device(
            p,
            input_device_name=None,
            input_device_index=999,
        )


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_input_device_invalid_name(
    mock_pyaudio_class: Mock,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test error handling for invalid device name."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    with audio.pyaudio_context() as p, pytest.raises(ValueError, match="No input device found"):
        audio.input_device(
            p,
            input_device_name="NonExistentDevice",
            input_device_index=None,
        )


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_output_device_invalid_name(
    mock_pyaudio_class: Mock,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test error handling for invalid output device name."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Try to get device with invalid name - should raise ValueError
    with audio.pyaudio_context() as p, pytest.raises(ValueError, match="No output device found"):
        audio.output_device(
            p,
            input_device_name="NonExistentOutputDevice",
            input_device_index=None,
        )


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_pyaudio_context_manager(
    mock_pyaudio_class: Mock,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test PyAudio context manager."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Test context manager
    with audio.pyaudio_context() as p:
        assert p is not None
        assert isinstance(p, MockPyAudio)

    # PyAudio should be terminated after context exits
    # (MockPyAudio doesn't track this, but real PyAudio would)


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_open_pyaudio_stream_context_manager(
    mock_pyaudio_class: Mock,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test PyAudio stream context manager."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    with audio.pyaudio_context() as p:
        # Test input stream
        with audio.open_pyaudio_stream(
            p,
            input=True,
            input_device_index=0,
        ) as stream:
            assert stream is not None
            assert stream.is_input
            assert not stream.is_output

        # Test output stream
        with audio.open_pyaudio_stream(
            p,
            output=True,
            input_device_index=1,
        ) as stream:
            assert stream is not None
            assert not stream.is_input
            assert stream.is_output


@patch("agent_cli.audio.pyaudio.PyAudio")
def test_device_filtering_by_capabilities(
    mock_pyaudio_class: Mock,
) -> None:
    """Test that devices are properly filtered by input/output capabilities."""
    # Create devices with specific capabilities
    device_info = [
        {"index": 0, "name": "Input Only", "maxInputChannels": 2, "maxOutputChannels": 0},
        {"index": 1, "name": "Output Only", "maxInputChannels": 0, "maxOutputChannels": 2},
        {"index": 2, "name": "Both", "maxInputChannels": 2, "maxOutputChannels": 2},
        {"index": 3, "name": "Neither", "maxInputChannels": 0, "maxOutputChannels": 0},
    ]

    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    with audio.pyaudio_context() as p:
        # Test input device filtering
        input_device_index, input_device_name = audio.input_device(
            p,
            input_device_name=None,
            input_device_index=0,
        )  # Input Only
        assert input_device_name == "Input Only"

        # Should skip "Output Only" and find "Both" for input
        mixed_input_index, mixed_input_name = audio.input_device(
            p,
            input_device_name=None,
            input_device_index=2,
        )  # Both
        assert mixed_input_name == "Both"

        # Test output device filtering
        output_device_index, output_device_name = audio.output_device(
            p,
            input_device_name=None,
            input_device_index=1,
        )  # Output Only
        assert output_device_name == "Output Only"

        mixed_output_index, mixed_output_name = audio.output_device(
            p,
            input_device_name=None,
            input_device_index=2,
        )  # Both
        assert mixed_output_name == "Both"
