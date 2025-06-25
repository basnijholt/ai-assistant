"""Module for Automatic Speech Recognition using Wyoming."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pyaudio
from wyoming.audio import AudioChunk, AudioStart, AudioStop

if TYPE_CHECKING:
    import logging
    from collections.abc import Generator

    from rich.console import Console
    from rich.live import Live
    from wyoming.client import AsyncClient

# PyAudio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 1024


@contextmanager
def pyaudio_context() -> Generator[pyaudio.PyAudio, None, None]:
    """Context manager for PyAudio lifecycle."""
    p = pyaudio.PyAudio()
    try:
        yield p
    finally:
        p.terminate()


@contextmanager
def open_pyaudio_stream(
    p: pyaudio.PyAudio,
    *args: object,
    **kwargs: object,
) -> Generator[pyaudio.Stream, None, None]:
    """Context manager for a PyAudio stream that ensures it's properly closed."""
    stream = p.open(*args, **kwargs)
    try:
        yield stream
    finally:
        stream.stop_stream()
        stream.close()


def list_input_devices(p: pyaudio.PyAudio, console: Console | None) -> None:
    """Print a numbered list of available input devices."""
    if console:
        console.print("[bold]Available input devices:[/bold]")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0 and console:
                console.print(f"  [yellow]{i}[/yellow]: {info['name']}")


async def send_audio(
    client: AsyncClient,
    stream: pyaudio.Stream,
    stop_event: asyncio.Event,
    logger: logging.Logger,
    live: Live | None,
) -> None:
    """Read from mic and send to Wyoming server."""
    await client.write_event(AudioStart(rate=RATE, width=2, channels=CHANNELS).event())

    try:
        seconds_streamed = 0.0
        while not stop_event.is_set():
            chunk = await asyncio.to_thread(
                stream.read,
                num_frames=CHUNK_SIZE,
                exception_on_overflow=False,
            )
            await client.write_event(
                AudioChunk(rate=RATE, width=2, channels=CHANNELS, audio=chunk).event(),
            )
            logger.debug("Sent %d byte(s) of audio", len(chunk))
            if live:
                seconds_streamed += len(chunk) / (RATE * CHANNELS * 2)
                from rich.text import Text

                live.update(
                    Text(f"Listening... ({seconds_streamed:.1f}s)", style="blue"),
                )
    finally:
        await client.write_event(AudioStop().event())
        logger.debug("Sent AudioStop")


async def receive_text(
    client: AsyncClient,
    logger: logging.Logger,
) -> str:
    """Receive transcription events and return the final transcript."""
    transcript_text = ""
    while True:
        event = await client.read_event()
        if event is None:
            logger.warning("Connection to ASR server lost.")
            break

        from wyoming.asr import Transcript, TranscriptChunk, TranscriptStart, TranscriptStop

        if Transcript.is_type(event.type):
            transcript = Transcript.from_event(event)
            transcript_text = transcript.text
            logger.info("Final transcript: %s", transcript_text)
            break
        if TranscriptChunk.is_type(event.type):
            chunk = TranscriptChunk.from_event(event)
            # This would print the partial transcript, we can pass a callback for this
            logger.debug("Transcript chunk: %s", chunk.text)
        elif TranscriptStart.is_type(event.type) or TranscriptStop.is_type(event.type):
            logger.debug("Received %s", event.type)
        else:
            logger.debug("Ignoring event type: %s", event.type)

    return transcript_text
