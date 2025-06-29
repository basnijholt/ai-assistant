"""Mock PyAudio for testing audio functionality without real hardware."""

from __future__ import annotations

import threading
import time
from typing import Any, Self


class MockAudioStream:
    """Mock audio stream for testing."""

    def __init__(
        self,
        *,
        is_input: bool = False,
        is_output: bool = False,
        simulate_delay: float = 0.0,
    ) -> None:
        """Initialize mock audio stream.

        Args:
            is_input: Whether this is an input stream
            is_output: Whether this is an output stream
            simulate_delay: Delay to simulate in operations

        """
        self.is_input = is_input
        self.is_output = is_output
        self.simulate_delay = min(simulate_delay, 0.01)  # Cap at 10ms for tests
        self.input_data: list[bytes] = []
        self.written_data: list[bytes] = []
        self.is_active = True
        self._lock = threading.Lock()

    def read(self, num_frames: int, *, exception_on_overflow: bool = True) -> bytes:  # noqa: ARG002
        """Simulate reading from audio input device."""
        if self.simulate_delay > 0:
            time.sleep(self.simulate_delay)

        # Generate synthetic audio data
        data = b"\x00\x01" * num_frames  # 16-bit audio data
        with self._lock:
            self.input_data.append(data)
        return data

    def write(self, frames: bytes) -> None:
        """Simulate writing to audio output device."""
        if self.simulate_delay > 0:
            time.sleep(self.simulate_delay)

        with self._lock:
            self.written_data.append(frames)

    def start_stream(self) -> None:
        """Start the mock stream."""
        self.is_active = True

    def stop_stream(self) -> None:
        """Stop the mock stream."""
        self.is_active = False

    def close(self) -> None:
        """Close the mock stream."""
        self.is_active = False

    def get_written_data(self) -> bytes:
        """Get all written data concatenated."""
        with self._lock:
            return b"".join(self.written_data)

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()


class MockPyAudio:
    """Mock PyAudio class for testing."""

    def __init__(self, device_info: list[dict[str, Any]]) -> None:
        """Initialize mock PyAudio with device information."""
        self.device_info = device_info
        self.streams: list[MockAudioStream] = []

    def get_device_count(self) -> int:
        """Get number of audio devices."""
        return len(self.device_info)

    def get_device_info_by_index(self, device_index: int) -> dict[str, Any]:
        """Get device info by index."""
        if 0 <= device_index < len(self.device_info):
            return self.device_info[device_index].copy()
        msg = f"Invalid device index: {device_index}"
        raise ValueError(msg)

    def get_format_from_width(self, width: int) -> str:
        """Get audio format from sample width."""
        format_map = {1: "paInt8", 2: "paInt16", 3: "paInt24", 4: "paInt32"}
        return format_map.get(width, "paInt16")

    def open(
        self,
        *,
        audio_format: str | None = None,  # noqa: ARG002
        channels: int = 1,  # noqa: ARG002
        rate: int = 44100,  # noqa: ARG002
        is_input: bool = False,  # Renamed from 'input'
        output: bool = False,
        input_device_index: int | None = None,  # noqa: ARG002
        output_device_index: int | None = None,  # noqa: ARG002
        frames_per_buffer: int = 1024,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> MockAudioStream:
        """Open a mock audio stream."""
        stream = MockAudioStream(
            is_input=is_input,
            is_output=output,
            simulate_delay=0.001,
        )
        self.streams.append(stream)
        return stream

    def terminate(self) -> None:
        """Terminate PyAudio."""
        for stream in self.streams:
            stream.close()

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.terminate()


def mock_pyaudio_context() -> MockPyAudio:
    """Create a mock PyAudio context for testing."""
    return MockPyAudio([])


def create_mock_audio_input(
    *,
    duration: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
) -> bytes:
    """Generate mock audio input data for testing."""
    samples = int(duration * sample_rate)
    # Generate simple sine wave pattern
    return b"\x00\x01" * samples * channels
