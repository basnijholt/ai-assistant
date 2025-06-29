"""Mock Wyoming servers and clients for testing."""

from __future__ import annotations

import asyncio
import io
import wave
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from wyoming.asr import Transcript, TranscriptChunk, TranscriptStart, TranscriptStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class MockWyomingEvent:
    """Mock Wyoming event for testing."""
    
    def __init__(self, event_type: str, data: dict | None = None) -> None:
        """Initialize mock event.
        
        Args:
            event_type: Type of the event
            data: Optional event data
            
        """
        self.type = event_type
        self.data = data or {}


class MockASRClient:
    """Mock Wyoming ASR client for testing transcription."""
    
    def __init__(
        self,
        *,
        transcript_responses: dict[str, str] | None = None,
        simulate_streaming: bool = True,
        simulate_delay: float = 0.1,
    ) -> None:
        """Initialize mock ASR client.
        
        Args:
            transcript_responses: Mapping of audio patterns to transcript responses
            simulate_streaming: Whether to simulate streaming transcript chunks
            simulate_delay: Delay between events
            
        """
        self.transcript_responses = transcript_responses or {"default": "mock transcription"}
        self.simulate_streaming = simulate_streaming
        self.simulate_delay = min(simulate_delay, 0.01)  # Cap at 10ms for tests
        self.received_audio_data = io.BytesIO()
        self.events_written: list[Event] = []
        self.is_active = True
    
    async def write_event(self, event: Event) -> None:
        """Mock writing an event to the server."""
        if not self.is_active:
            return
        
        self.events_written.append(event)
        
        # Simulate receiving audio data
        if event.type == "audio-chunk":
            chunk_data = event.data.get("audio", b"")
            self.received_audio_data.write(chunk_data)
        
        # Add small delay to simulate network
        if self.simulate_delay > 0:
            await asyncio.sleep(self.simulate_delay * 0.1)
    
    async def read_event(self) -> Event | None:
        """Mock reading events from the server."""
        if not self.is_active:
            return None
        
        # Check if we have a generator for streaming events
        if not hasattr(self, '_event_generator'):
            # Create generator based on received audio
            audio_data = self.received_audio_data.getvalue()
            transcript_text = self._determine_transcript(audio_data)
            
            if self.simulate_streaming:
                self._event_generator = self._generate_streaming_events(transcript_text)
            else:
                # Simple non-streaming response
                async def simple_response():
                    yield Transcript(text=transcript_text).event()
                self._event_generator = simple_response()
        
        # Return next event from generator
        try:
            return await self._event_generator.__anext__()
        except StopAsyncIteration:
            return None
    
    def _determine_transcript(self, audio_data: bytes) -> str:
        """Determine transcript text based on audio data pattern."""
        if not audio_data:
            return "mock transcription"
        
        # Use a simple heuristic based on audio data pattern
        pattern = audio_data[0] if audio_data else 0
        
        # Map patterns to responses
        for key, response in self.transcript_responses.items():
            if key == "default":
                continue
            # Simple mapping based on first byte
            if pattern == (ord(key[0]) if key else 0):
                return response
        
        return self.transcript_responses.get("default", "mock transcription")
    
    async def _generate_streaming_events(self, final_text: str) -> AsyncGenerator[Event, None]:
        """Generate streaming transcript events."""
        # Start event
        yield TranscriptStart().event()
        if self.simulate_delay > 0:
            await asyncio.sleep(min(self.simulate_delay, 0.1))  # Cap delay for tests
        
        # Chunk events (split text into words)
        words = final_text.split()
        current_text = ""
        
        for word in words:
            current_text += (word + " ")
            yield TranscriptChunk(text=current_text.strip()).event()
            if self.simulate_delay > 0:
                await asyncio.sleep(min(self.simulate_delay * 0.1, 0.05))  # Very short delays for tests
        
        # Stop event
        yield TranscriptStop().event()
        if self.simulate_delay > 0:
            await asyncio.sleep(min(self.simulate_delay, 0.1))
        
        # Final transcript
        yield Transcript(text=final_text).event()
    
    async def __aenter__(self) -> MockASRClient:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        self.is_active = False


class MockTTSClient:
    """Mock Wyoming TTS client for testing speech synthesis."""
    
    def __init__(
        self,
        *,
        audio_responses: dict[str, bytes] | None = None,
        simulate_delay: float = 0.1,
    ) -> None:
        """Initialize mock TTS client.
        
        Args:
            audio_responses: Mapping of text patterns to audio responses
            simulate_delay: Delay between events
            
        """
        self.audio_responses = audio_responses or {}
        self.simulate_delay = min(simulate_delay, 0.01)  # Cap at 10ms for tests
        self.events_written: list[Event] = []
        self.synthesis_text = ""
        self.is_active = True
    
    async def write_event(self, event: Event) -> None:
        """Mock writing an event to the server."""
        if not self.is_active:
            return
        
        self.events_written.append(event)
        
        # Extract synthesis text
        if event.type == "synthesize":
            self.synthesis_text = event.data.get("text", "")
        
        # Add small delay to simulate network
        if self.simulate_delay > 0:
            await asyncio.sleep(self.simulate_delay * 0.1)
    
    async def read_event(self) -> Event | None:
        """Mock reading events from the server."""
        if not self.is_active:
            return None
        
        # Check if we have a generator for audio events
        if not hasattr(self, '_event_generator'):
            self._event_generator = self._generate_audio_events()
        
        # Return next event from generator
        try:
            return await self._event_generator.__anext__()
        except StopAsyncIteration:
            return None
    
    async def _generate_audio_events(self) -> AsyncGenerator[Event, None]:
        """Generate audio synthesis events."""
        # Get or create audio data
        audio_data = self._get_audio_for_text(self.synthesis_text)
        
        # Audio start event
        yield AudioStart(rate=22050, width=2, channels=1).event()
        
        # Split audio into chunks
        chunk_size = 4096
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            yield AudioChunk(
                rate=22050,
                width=2,
                channels=1,
                audio=chunk,
            ).event()
            if self.simulate_delay > 0:
                await asyncio.sleep(self.simulate_delay * 0.2)
        
        # Audio stop event
        yield AudioStop().event()
    
    def _get_audio_for_text(self, text: str) -> bytes:
        """Get audio data for the given text."""
        # Check if we have a specific response for this text
        for pattern, audio_data in self.audio_responses.items():
            if pattern.lower() in text.lower():
                return audio_data
        
        # Generate synthetic audio based on text length
        return self._generate_synthetic_audio(text)
    
    def _generate_synthetic_audio(self, text: str) -> bytes:
        """Generate synthetic audio data for testing."""
        # Create simple audio based on text length
        sample_rate = 22050
        duration = max(0.5, len(text) * 0.05)  # 50ms per character, min 0.5s
        samples = int(sample_rate * duration)
        
        # Create a simple pattern based on text hash
        pattern = hash(text) % 256
        audio_frames = bytes([pattern, pattern ^ 255] * samples)
        
        return audio_frames
    
    async def __aenter__(self) -> MockTTSClient:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        self.is_active = False


class MockWyomingAsyncClient:
    """Mock Wyoming AsyncClient that can act as either ASR or TTS client."""
    
    _asr_responses: dict[str, str] = {}
    _tts_responses: dict[str, bytes] = {}
    _simulate_delay: float = 0.1
    
    @classmethod
    def from_uri(
        cls,
        uri: str,
        *,
        asr_responses: dict[str, str] | None = None,
        tts_responses: dict[str, bytes] | None = None,
        simulate_delay: float = 0.1,
    ) -> MockASRClient | MockTTSClient:
        """Create appropriate mock client based on URI."""
        # Store configuration
        if asr_responses is not None:
            cls._asr_responses = asr_responses
        if tts_responses is not None:
            cls._tts_responses = tts_responses
        cls._simulate_delay = simulate_delay
        
        # Determine client type based on URI/port
        if ":10300" in uri or "asr" in uri.lower():
            return MockASRClient(
                transcript_responses=cls._asr_responses,
                simulate_delay=cls._simulate_delay,
            )
        elif ":10200" in uri or "tts" in uri.lower():
            return MockTTSClient(
                audio_responses=cls._tts_responses,
                simulate_delay=cls._simulate_delay,
            )
        else:
            # Default to ASR
            return MockASRClient(
                transcript_responses=cls._asr_responses,
                simulate_delay=cls._simulate_delay,
            )


def create_mock_audio_data(text: str, sample_rate: int = 22050) -> bytes:
    """Create mock audio data for TTS testing.
    
    Args:
        text: Text that this audio represents
        sample_rate: Audio sample rate
        
    Returns:
        Raw audio data as bytes
        
    """
    duration = max(0.5, len(text) * 0.05)  # 50ms per character
    samples = int(sample_rate * duration)
    
    # Create pattern based on text
    pattern = hash(text) % 128
    audio_data = bytes([pattern, 255 - pattern] * samples)
    
    return audio_data 