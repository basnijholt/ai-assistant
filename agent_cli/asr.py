"""Module for Automatic Speech Recognition using Wyoming."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pyaudio
from rich.text import Text
from wyoming.asr import Transcribe, Transcript, TranscriptChunk, TranscriptStart, TranscriptStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient

from agent_cli import config

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable, Generator

    from rich.console import Console
    from rich.live import Live


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
    *,
    live: Live | None = None,
) -> None:
    """Read from mic and send to Wyoming server.

    Args:
        client: Wyoming client connection
        stream: PyAudio stream
        stop_event: Event to stop recording
        logger: Logger instance
        live: Rich Live display for progress (transcribe mode)
        console: Rich Console for manual timing display (voice-assistant mode)

    """
    await client.write_event(Transcribe().event())

    await client.write_event(
        AudioStart(rate=config.PYAUDIO_RATE, width=2, channels=config.PYAUDIO_CHANNELS).event(),
    )

    try:
        seconds_streamed = 0.0
        while not stop_event.is_set():
            chunk = await asyncio.to_thread(
                stream.read,
                num_frames=config.PYAUDIO_CHUNK_SIZE,
                exception_on_overflow=False,
            )
            await client.write_event(
                AudioChunk(
                    rate=config.PYAUDIO_RATE,
                    width=2,
                    channels=config.PYAUDIO_CHANNELS,
                    audio=chunk,
                ).event(),
            )
            logger.debug("Sent %d byte(s) of audio", len(chunk))

            # Update display timing
            seconds_streamed += len(chunk) / (config.PYAUDIO_RATE * config.PYAUDIO_CHANNELS * 2)
            if live:
                live.update(Text(f"Listening... ({seconds_streamed:.1f}s)", style="blue"))

    finally:
        await client.write_event(AudioStop().event())
        logger.debug("Sent AudioStop")


async def receive_text(
    client: AsyncClient,
    logger: logging.Logger,
    *,
    chunk_callback: Callable[[str], None] | None = None,
    final_callback: Callable[[str], None] | None = None,
) -> str:
    """Receive transcription events and return the final transcript.

    Args:
        client: Wyoming client connection
        logger: Logger instance
        chunk_callback: Optional callback for transcript chunks (live partial results)
        final_callback: Optional callback for final transcript formatting

    """
    transcript_text = ""
    while True:
        event = await client.read_event()
        if event is None:
            logger.warning("Connection to ASR server lost.")
            break

        if Transcript.is_type(event.type):
            transcript = Transcript.from_event(event)
            transcript_text = transcript.text
            logger.info("Final transcript: %s", transcript_text)
            if final_callback:
                final_callback(transcript_text)
            break
        if TranscriptChunk.is_type(event.type):
            chunk = TranscriptChunk.from_event(event)
            logger.debug("Transcript chunk: %s", chunk.text)
            if chunk_callback:
                chunk_callback(chunk.text)
        elif TranscriptStart.is_type(event.type) or TranscriptStop.is_type(event.type):
            logger.debug("Received %s", event.type)
        else:
            logger.debug("Ignoring event type: %s", event.type)

    return transcript_text


async def transcribe_audio(
    asr_server_ip: str,
    asr_server_port: int,
    device_index: int | None,
    logger: logging.Logger,
    p: pyaudio.PyAudio,
    stop_event: asyncio.Event,
    *,
    console: Console | None = None,
    live: Live | None = None,
    listening_message: str = "Listening...",
    chunk_callback: Callable[[str], None] | None = None,
    final_callback: Callable[[str], None] | None = None,
) -> str | None:
    """Unified ASR transcription function for both transcribe and voice-assistant.

    Args:
        asr_server_ip: Wyoming server IP
        asr_server_port: Wyoming server port
        device_index: Audio input device index
        logger: Logger instance
        p: PyAudio instance
        stop_event: Event to stop recording
        console: Rich console for messages
        live: Rich Live display for progress
        listening_message: Message to display when starting
        chunk_callback: Callback for transcript chunks
        final_callback: Callback for final transcript

    Returns:
        Transcribed text or None if error

    """
    uri = f"tcp://{asr_server_ip}:{asr_server_port}"
    logger.info("Connecting to Wyoming server at %s", uri)

    try:
        async with AsyncClient.from_uri(uri) as client:
            logger.info("ASR connection established")
            if console:
                console.print(f"[green]{listening_message}[/green]")

            with open_pyaudio_stream(
                p,
                format=config.PYAUDIO_FORMAT,
                channels=config.PYAUDIO_CHANNELS,
                rate=config.PYAUDIO_RATE,
                input=True,
                frames_per_buffer=config.PYAUDIO_CHUNK_SIZE,
                input_device_index=device_index,
            ) as stream:
                send_task = asyncio.create_task(
                    send_audio(
                        client,
                        stream,
                        stop_event,
                        logger,
                        live=live,
                    ),
                )
                recv_task = asyncio.create_task(
                    receive_text(
                        client,
                        logger,
                        chunk_callback=chunk_callback,
                        final_callback=final_callback,
                    ),
                )

                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.ALL_COMPLETED,
                )
                for task in pending:
                    task.cancel()

                return recv_task.result()

    except ConnectionRefusedError:
        if console:
            console.print(
                f"[bold red]ASR Connection refused.[/bold red] Is the server at {uri} running?",
            )
        return None
    except Exception as e:
        logger.exception("An error occurred during transcription.")
        if console:
            console.print(f"[bold red]Transcription error:[/bold red] {e}")
        return None


def input_device(
    p: pyaudio.PyAudio,
    device_name: str | None,
    device_index: int | None,
) -> tuple[int | None, str | None]:
    """Find an input device by a prioritized, comma-separated list of keywords."""
    if device_name is None and device_index is None:
        return None, None

    if device_index is not None:
        info = p.get_device_info_by_index(device_index)
        return device_index, info.get("name")
    assert device_name is not None
    search_terms = [term.strip().lower() for term in device_name.split(",") if term.strip()]

    if not search_terms:
        msg = "Device name string is empty or contains only whitespace."
        raise ValueError(msg)

    input_devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        device_info_name = info.get("name")
        if device_info_name:
            input_devices.append((i, device_info_name))

    for term in search_terms:
        for index, name in input_devices:
            if term in name.lower():
                return index, name

    msg = f"No input device found matching any of the keywords in {device_name!r}"
    raise ValueError(msg)
