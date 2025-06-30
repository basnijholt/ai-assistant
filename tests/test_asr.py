"""Unit tests for the asr module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from wyoming.asr import Transcribe, Transcript, TranscriptChunk
from wyoming.audio import AudioChunk, AudioStart, AudioStop

from agent_cli import asr


@pytest.mark.asyncio
async def test_send_audio() -> None:
    """Test that send_audio sends the correct events."""
    # Arrange
    client = AsyncMock()
    stream = MagicMock()
    stop_event = asyncio.Event()

    def read_and_stop(*args, **kwargs) -> bytes:  # noqa: ARG001
        # This function will be called by stream.read()
        # It stops the loop after the first chunk.
        stop_event.set()
        return b"fake_audio_chunk"

    stream.read.side_effect = read_and_stop
    logger = MagicMock()

    # Act
    # No need to create a task and sleep, just await the coroutine.
    # The side_effect will stop the loop.
    await asr.send_audio(client, stream, stop_event, logger)

    # Assert
    assert client.write_event.call_count == 4
    client.write_event.assert_any_call(Transcribe().event())
    client.write_event.assert_any_call(
        AudioStart(rate=16000, width=2, channels=1).event(),
    )
    client.write_event.assert_any_call(
        AudioChunk(
            rate=16000,
            width=2,
            channels=1,
            audio=b"fake_audio_chunk",
        ).event(),
    )
    client.write_event.assert_any_call(AudioStop().event())


@pytest.mark.asyncio
async def test_receive_text() -> None:
    """Test that receive_text correctly processes events."""
    # Arrange
    client = AsyncMock()
    client.read_event.side_effect = [
        TranscriptChunk(text="hello").event(),
        Transcript(text="hello world").event(),
        None,  # To stop the loop
    ]
    logger = MagicMock()
    chunk_callback = MagicMock()
    final_callback = MagicMock()

    # Act
    result = await asr.receive_text(
        client,
        logger,
        chunk_callback=chunk_callback,
        final_callback=final_callback,
    )

    # Assert
    assert result == "hello world"
    chunk_callback.assert_called_once_with("hello")
    final_callback.assert_called_once_with("hello world")


@pytest.mark.asyncio
async def test_transcribe_audio() -> None:
    """Test the main transcribe_audio function."""
    # Arrange
    with (
        patch("agent_cli.asr.AsyncClient.from_uri") as mock_from_uri,
        patch(
            "agent_cli.audio.pyaudio_context",
        ) as mock_pyaudio_context,
    ):
        mock_client = AsyncMock()
        mock_client.read_event.side_effect = [
            Transcript(text="test transcription").event(),
            None,
        ]
        mock_from_uri.return_value.__aenter__.return_value = mock_client

        p = MagicMock()
        mock_pyaudio_context.return_value.__enter__.return_value = p
        stream = MagicMock()
        p.open.return_value.__enter__.return_value = stream
        stop_event = asyncio.Event()
        logger = MagicMock()

        # Act
        transcribe_task = asyncio.create_task(
            asr.transcribe_audio(
                "localhost",
                12345,
                0,
                logger,
                p,
                stop_event,
                console=MagicMock(),
            ),
        )
        # Give the task a moment to start up
        await asyncio.sleep(0.1)
        # Set the stop event to allow the send_audio task to complete
        stop_event.set()
        # Await the result of the transcription
        result = await transcribe_task

        # Assert
        assert result == "test transcription"


@pytest.mark.asyncio
async def test_transcribe_audio_connection_error() -> None:
    """Test the main transcribe_audio function with a connection error."""
    # Arrange
    with (
        patch(
            "agent_cli.asr.AsyncClient.from_uri",
            side_effect=ConnectionRefusedError,
        ),
        patch("agent_cli.audio.pyaudio_context") as mock_pyaudio_context,
    ):
        p = MagicMock()
        mock_pyaudio_context.return_value.__enter__.return_value = p
        stream = MagicMock()
        p.open.return_value.__enter__.return_value = stream
        stop_event = asyncio.Event()
        logger = MagicMock()

        # Act
        result = await asr.transcribe_audio(
            "localhost",
            12345,
            0,
            logger,
            p,
            stop_event,
            console=MagicMock(),
        )

        # Assert
        assert result is None
