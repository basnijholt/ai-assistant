"""Mock PyAudio for testing audio functionality without real hardware."""

from __future__ import annotations

from typing import Any, Self


class MockAudioStream:
    """Mock audio stream for testing."""

    def __init__(self, *, is_input: bool = False, is_output: bool = False) -> None:
        """Initialize mock audio stream."""
        self.is_input = is_input
        self.is_output = is_output
        self.written_data: list[bytes] = []
        self.is_active = True

    def read(self, num_frames: int, *, exception_on_overflow: bool = True) -> bytes:  # noqa: ARG002
        """Simulate reading from audio input device."""
        return b"\x00\x01" * num_frames  # 16-bit audio data

    def write(self, frames: bytes) -> None:
        """Simulate writing to audio output device."""
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
            return self.device_info[device_index]
        msg = f"Invalid device index: {device_index}"
        raise ValueError(msg)

    def get_format_from_width(self, width: int) -> str:
        """Get audio format from sample width."""
        format_map = {1: "paInt8", 2: "paInt16", 3: "paInt24", 4: "paInt32"}
        return format_map.get(width, "paInt16")

    def open(
        self,
        **kwargs: Any,
    ) -> MockAudioStream:
        """Open a mock audio stream."""
        stream = MockAudioStream(
            is_input=kwargs.get("input", False),
            is_output=kwargs.get("output", False),
        )
        self.streams.append(stream)
        return stream

    def terminate(self) -> None:
        """Terminate PyAudio."""

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.terminate()
