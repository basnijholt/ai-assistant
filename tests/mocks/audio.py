"""Mock audio devices and PyAudio for testing."""

from __future__ import annotations

import io
import threading
import time
import wave


class MockAudioStream:
    """Mock PyAudio stream with controllable behavior."""

    def __init__(
        self,
        *,
        input_data: bytes | None = None,
        simulate_delay: float = 0.01,
        chunk_size: int = 1024,
    ) -> None:
        """Initialize mock audio stream.

        Args:
            input_data: Pre-recorded audio data for input streams
            simulate_delay: Delay between read operations
            chunk_size: Size of audio chunks

        """
        self.input_data = input_data or b"\x00\x00" * 1024  # Default silence
        self.simulate_delay = simulate_delay
        self.chunk_size = chunk_size
        self.position = 0
        self.is_active = True
        self.output_buffer = io.BytesIO()
        self._lock = threading.Lock()

    def read(self, num_frames: int, *, exception_on_overflow: bool = True) -> bytes:
        """Simulate reading from audio input device."""
        if self.simulate_delay > 0:
            time.sleep(self.simulate_delay)

        with self._lock:
            if not self.is_active:
                return b""

            # Calculate bytes needed (2 bytes per frame for 16-bit audio)
            bytes_needed = num_frames * 2

            # Return chunk from input data, cycling if necessary
            if self.position >= len(self.input_data):
                self.position = 0

            chunk = self.input_data[self.position : self.position + bytes_needed]
            if len(chunk) < bytes_needed:
                # Pad with zeros if we don't have enough data
                chunk += b"\x00" * (bytes_needed - len(chunk))

            self.position += len(chunk)
            return chunk

    def write(self, frames: bytes) -> None:
        """Simulate writing to audio output device."""
        if self.simulate_delay > 0:
            time.sleep(self.simulate_delay)

        with self._lock:
            if self.is_active:
                self.output_buffer.write(frames)

    def get_written_data(self) -> bytes:
        """Get all data written to the output stream."""
        with self._lock:
            return self.output_buffer.getvalue()

    def start_stream(self) -> None:
        """Start the stream."""
        self.is_active = True

    def stop_stream(self) -> None:
        """Stop the stream."""
        self.is_active = False

    def close(self) -> None:
        """Close the stream."""
        self.is_active = False

    def __enter__(self) -> MockAudioStream:
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()


class MockPyAudio:
    """Mock PyAudio instance with controllable devices and streams."""

    def __init__(self, device_info: list[dict] | None = None) -> None:
        """Initialize mock PyAudio.

        Args:
            device_info: List of mock device information

        """
        self.device_info = device_info or []
        self.streams: list[MockAudioStream] = []
        self._format_map = {
            1: "int8",
            2: "int16",
            4: "int32",
        }

    def get_device_count(self) -> int:
        """Get number of audio devices."""
        return len(self.device_info)

    def get_device_info_by_index(self, device_index: int) -> dict:
        """Get device info by index."""
        if 0 <= device_index < len(self.device_info):
            return self.device_info[device_index].copy()
        raise ValueError(f"Invalid device index: {device_index}")

    def get_format_from_width(self, width: int) -> str:
        """Get format from sample width."""
        return self._format_map.get(width, "unknown")

    def open(
        self,
        *,
        format: str | None = None,
        channels: int = 1,
        rate: int = 44100,
        input: bool = False,
        output: bool = False,
        input_device_index: int | None = None,
        output_device_index: int | None = None,
        frames_per_buffer: int = 1024,
        **kwargs,
    ) -> MockAudioStream:
        """Open a mock audio stream."""
        # Generate some mock input data if this is an input stream
        input_data = None
        if input:
            # Create 5 seconds of mock audio data
            duration = 5.0
            samples = int(rate * duration)
            input_data = b"\x00\x01" * samples  # Simple pattern

        stream = MockAudioStream(
            input_data=input_data,
            chunk_size=frames_per_buffer,
        )
        self.streams.append(stream)
        return stream

    def terminate(self) -> None:
        """Terminate PyAudio."""
        for stream in self.streams:
            stream.close()
        self.streams.clear()


def create_mock_audio_input(text_to_simulate: str = "test") -> bytes:
    """Create mock audio data that would transcribe to the given text.

    This is a placeholder - in reality, we'd need actual audio data
    that corresponds to the text, but for testing we just return
    synthetic data and rely on the Wyoming mock to return the expected text.

    Args:
        text_to_simulate: Text that this audio should transcribe to

    Returns:
        Mock audio data as bytes

    """
    # Create simple WAV data (silence that our mock will "transcribe" to the text)
    sample_rate = 16000
    duration = len(text_to_simulate) * 0.1  # 100ms per character
    samples = int(sample_rate * duration)

    # Simple pattern that varies with text length
    pattern = (text_to_simulate.encode("utf-8")[0] if text_to_simulate else 0) % 256
    audio_frames = bytes([pattern, 0] * samples)

    # Create WAV data
    wav_data = io.BytesIO()
    with wave.open(wav_data, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_frames)

    return wav_data.getvalue()


def mock_pyaudio_context(device_info: list[dict] | None = None) -> MockPyAudio:
    """Create a mock PyAudio context for testing.

    Args:
        device_info: Optional list of mock device information

    Returns:
        Mock PyAudio instance

    """
    return MockPyAudio(device_info)
