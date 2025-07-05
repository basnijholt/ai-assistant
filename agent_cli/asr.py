"""Module for Automatic Speech Recognition using Wyoming."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from wyoming.asr import Transcribe, Transcript, TranscriptChunk, TranscriptStart, TranscriptStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop

from agent_cli import config
from agent_cli.audio import open_pyaudio_stream, read_audio_stream, setup_input_stream
from agent_cli.wyoming_utils import manage_send_receive_tasks, wyoming_client_context

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable

    import pyaudio
    from rich.live import Live
    from wyoming.client import AsyncClient

    from agent_cli.utils import InteractiveStopEvent


async def send_audio(
    client: AsyncClient,
    stream: pyaudio.Stream,
    stop_event: InteractiveStopEvent,
    logger: logging.Logger,
    *,
    live: Live,
    quiet: bool = False,
) -> None:
    """Read from mic and send to Wyoming server.

    Args:
        client: Wyoming client connection
        stream: PyAudio stream
        stop_event: Event to stop recording
        logger: Logger instance
        live: Rich Live display for progress (transcribe mode)
        quiet: If True, suppress all console output

    """
    await client.write_event(Transcribe().event())
    await client.write_event(AudioStart(**config.WYOMING_AUDIO_CONFIG).event())

    async def send_chunk(chunk: bytes) -> None:
        """Send audio chunk to ASR server."""
        await client.write_event(
            AudioChunk(audio=chunk, **config.WYOMING_AUDIO_CONFIG).event(),
        )

    try:
        # Use common audio reading function
        await read_audio_stream(
            stream=stream,
            stop_event=stop_event,
            chunk_handler=send_chunk,
            logger=logger,
            live=live,
            quiet=quiet,
            progress_message="Listening",
            progress_style="blue",
        )
    finally:
        await client.write_event(AudioStop().event())
        logger.debug("Sent AudioStop")


async def record_audio_to_buffer(
    stream: pyaudio.Stream,
    stop_event: InteractiveStopEvent,
    logger: logging.Logger,
    *,
    live: Live | None = None,
    quiet: bool = False,
    progress_message: str = "Recording",
    progress_style: str = "blue",
) -> bytes:
    """Record audio from mic to buffer.

    Args:
        stream: PyAudio stream
        stop_event: Event to stop recording
        logger: Logger instance
        live: Rich Live display for progress (optional)
        quiet: If True, suppress all console output
        progress_message: Message to display during recording
        progress_style: Rich style for progress messages

    Returns:
        Raw audio data as bytes

    """
    audio_buffer = io.BytesIO()

    def buffer_chunk(chunk: bytes) -> None:
        """Buffer audio chunk."""
        audio_buffer.write(chunk)

    # Use common audio reading function
    await read_audio_stream(
        stream=stream,
        stop_event=stop_event,
        chunk_handler=buffer_chunk,
        logger=logger,
        live=live,
        quiet=quiet,
        progress_message=progress_message,
        progress_style=progress_style,
    )

    return audio_buffer.getvalue()


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
    input_device_index: int | None,
    logger: logging.Logger,
    p: pyaudio.PyAudio,
    stop_event: InteractiveStopEvent,
    *,
    live: Live,
    quiet: bool = False,
    chunk_callback: Callable[[str], None] | None = None,
    final_callback: Callable[[str], None] | None = None,
) -> str | None:
    """Unified ASR transcription function for both transcribe and voice-assistant.

    Args:
        asr_server_ip: Wyoming server IP
        asr_server_port: Wyoming server port
        input_device_index: Audio input device index
        logger: Logger instance
        p: PyAudio instance
        stop_event: Event to stop recording
        live: Rich Live display for progress
        quiet: If True, suppress all console output
        chunk_callback: Callback for transcript chunks
        final_callback: Callback for final transcript

    Returns:
        Transcribed text or None if error

    """
    try:
        async with wyoming_client_context(
            asr_server_ip,
            asr_server_port,
            "ASR",
            logger,
            quiet=quiet,
        ) as client:
            stream_config = setup_input_stream(input_device_index)
            with open_pyaudio_stream(p, **stream_config) as stream:
                send_task, recv_task = await manage_send_receive_tasks(
                    send_audio(client, stream, stop_event, logger, live=live, quiet=quiet),
                    receive_text(
                        client,
                        logger,
                        chunk_callback=chunk_callback,
                        final_callback=final_callback,
                    ),
                )
                return recv_task.result()
    except (ConnectionRefusedError, Exception):
        return None
