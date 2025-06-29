"""Shared test fixtures and configuration."""

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator


@pytest.fixture
def mock_console() -> Console:
    """Provide a console that writes to a StringIO for testing."""
    return Console(file=io.StringIO(), width=80, force_terminal=True)


@pytest.fixture
def mock_logger() -> logging.Logger:
    """Provide a mock logger for testing."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def stop_event() -> asyncio.Event:
    """Provide an asyncio event for stopping operations."""
    return asyncio.Event()


@pytest.fixture(autouse=True)
def setup_test_timeouts() -> None:
    """Automatically set up timeouts for all tests."""
    # Set a default timeout for asyncio operations
    import asyncio
    # This will affect all asyncio operations in tests
    asyncio.get_event_loop().set_debug(True)


@pytest.fixture
def timeout_seconds() -> float:
    """Default timeout for async operations in tests."""
    return 5.0


@pytest.fixture
def synthetic_audio_data() -> bytes:
    """Generate synthetic WAV audio data for testing."""
    # Create a simple sine wave audio data
    sample_rate = 16000
    duration = 1.0  # 1 second
    samples = int(sample_rate * duration)
    
    # Generate simple audio data (silence for simplicity)
    audio_frames = b'\x00\x00' * samples  # 16-bit silence
    
    # Create WAV data
    wav_data = io.BytesIO()
    with wave.open(wav_data, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_frames)
    
    return wav_data.getvalue()


@pytest.fixture
def mock_pyaudio_device_info() -> list[dict]:
    """Mock PyAudio device info for testing."""
    return [
        {
            "index": 0,
            "name": "Mock Input Device",
            "maxInputChannels": 2,
            "maxOutputChannels": 0,
            "defaultSampleRate": 44100.0,
        },
        {
            "index": 1,
            "name": "Mock Output Device",
            "maxInputChannels": 0,
            "maxOutputChannels": 2,
            "defaultSampleRate": 44100.0,
        },
        {
            "index": 2,
            "name": "Mock Combined Device",
            "maxInputChannels": 2,
            "maxOutputChannels": 2,
            "defaultSampleRate": 44100.0,
        },
    ]


@pytest.fixture
def transcript_responses() -> dict[str, str]:
    """Predefined transcript responses for testing."""
    return {
        "hello": "Hello there!",
        "test": "This is a test transcription.",
        "question": "What is the meaning of life?",
        "long": "This is a longer transcription that spans multiple words and should test the chunking functionality properly.",
    }


@pytest.fixture
def llm_responses() -> dict[str, str]:
    """Predefined LLM responses for testing."""
    return {
        "correct": "This text has been corrected and improved.",
        "hello": "Hello! How can I help you today?",
        "question": "The meaning of life is 42, according to The Hitchhiker's Guide to the Galaxy.",
        "default": "I understand your request and here is my response.",
    } 