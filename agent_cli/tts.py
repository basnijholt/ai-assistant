"""Module for Text-to-Speech using Wyoming."""

from __future__ import annotations

import io
import wave
from typing import TYPE_CHECKING

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.tts import Synthesize

from agent_cli import config
from agent_cli.asr import (
    open_pyaudio_stream,
    pyaudio_context,
)
from agent_cli.utils import print_error_message, print_status_message

if TYPE_CHECKING:
    import logging

    from rich.console import Console


async def synthesize_speech(
    text: str,
    tts_server_ip: str,
    tts_server_port: int,
    logger: logging.Logger,
    *,
    voice_name: str | None = None,
    language: str | None = None,
    speaker: str | None = None,
    console: Console | None = None,
) -> bytes | None:
    """Synthesize speech from text using Wyoming TTS server.

    Args:
        text: Text to synthesize
        tts_server_ip: Wyoming TTS server IP
        tts_server_port: Wyoming TTS server port
        logger: Logger instance
        voice_name: Optional voice name
        language: Optional language
        speaker: Optional speaker name
        console: Rich console for messages

    Returns:
        WAV audio data as bytes, or None if error

    """
    uri = f"tcp://{tts_server_ip}:{tts_server_port}"
    logger.info("Connecting to Wyoming TTS server at %s", uri)

    try:
        async with AsyncClient.from_uri(uri) as client:
            logger.info("TTS connection established")
            if console:
                print_status_message(console, f"🔊 Synthesizing: {text[:50]}...")

            # Create synthesize request
            synthesize_event = Synthesize(text=text)

            # Add voice parameters if specified
            if voice_name or language or speaker:
                voice_data = {}
                if voice_name:
                    voice_data["name"] = voice_name
                if language:
                    voice_data["language"] = language
                if speaker:
                    voice_data["speaker"] = speaker
                synthesize_event.data["voice"] = voice_data

            await client.write_event(synthesize_event.event())

            # Collect audio data
            audio_data = io.BytesIO()
            sample_rate = None
            sample_width = None
            channels = None

            while True:
                event = await client.read_event()
                if event is None:
                    logger.warning("Connection to TTS server lost.")
                    break

                if AudioStart.is_type(event.type):
                    audio_start = AudioStart.from_event(event)
                    sample_rate = audio_start.rate
                    sample_width = audio_start.width
                    channels = audio_start.channels
                    logger.debug(
                        "Audio stream started: %dHz, %d channels, %d bytes/sample",
                        sample_rate,
                        channels,
                        sample_width,
                    )

                elif AudioChunk.is_type(event.type):
                    chunk = AudioChunk.from_event(event)
                    audio_data.write(chunk.audio)
                    logger.debug("Received %d bytes of audio", len(chunk.audio))

                elif AudioStop.is_type(event.type):
                    logger.debug("Audio stream completed")
                    break
                else:
                    logger.debug("Ignoring event type: %s", event.type)

            # Convert to WAV format
            if sample_rate and sample_width and channels and audio_data.tell() > 0:
                wav_data = io.BytesIO()
                with wave.open(wav_data, "wb") as wav_file:
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(sample_width)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data.getvalue())

                logger.info("Speech synthesis completed: %d bytes", wav_data.tell())
                return wav_data.getvalue()
            logger.warning("No audio data received from TTS server")
            return None

    except ConnectionRefusedError:
        print_error_message(
            console,
            "TTS Connection refused.",
            f"Is the Wyoming TTS server running at {uri}?",
        )
        return None
    except Exception as e:
        logger.exception("An error occurred during speech synthesis.")
        print_error_message(console, f"TTS error: {e}")
        return None


async def play_audio(
    audio_data: bytes,
    logger: logging.Logger,
    *,
    output_device_index: int | None = None,
    console: Console | None = None,
) -> None:
    """Play WAV audio data using PyAudio with proper resource management.

    Args:
        audio_data: WAV audio data as bytes
        logger: Logger instance
        output_device_index: Optional output device index
        console: Rich console for messages

    """
    try:
        if console:
            print_status_message(console, "🔊 Playing audio...")

        # Parse WAV file
        wav_io = io.BytesIO(audio_data)
        with wave.open(wav_io, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())

        with (
            pyaudio_context() as p,
            open_pyaudio_stream(
                p,
                format=p.get_format_from_width(sample_width),
                channels=channels,
                rate=sample_rate,
                output=True,
                frames_per_buffer=config.PYAUDIO_CHUNK_SIZE,
                output_device_index=output_device_index,
            ) as stream,
        ):
            # Play in chunks to avoid blocking
            chunk_size = config.PYAUDIO_CHUNK_SIZE
            for i in range(0, len(frames), chunk_size):
                chunk = frames[i : i + chunk_size]
                stream.write(chunk)

        logger.info("Audio playback completed")
        if console:
            print_status_message(console, "✅ Audio playback finished")

    except Exception as e:
        logger.exception("Error during audio playback")
        print_error_message(console, f"Playback error: {e}")


async def speak_text(
    text: str,
    tts_server_ip: str,
    tts_server_port: int,
    logger: logging.Logger,
    *,
    voice_name: str | None = None,
    language: str | None = None,
    speaker: str | None = None,
    output_device_index: int | None = None,
    console: Console | None = None,
    play_audio_flag: bool = True,
) -> bytes | None:
    """Synthesize and optionally play speech from text.

    Args:
        text: Text to synthesize and speak
        tts_server_ip: Wyoming TTS server IP
        tts_server_port: Wyoming TTS server port
        logger: Logger instance
        voice_name: Optional voice name
        language: Optional language
        speaker: Optional speaker name
        output_device_index: Optional output device index
        console: Rich console for messages
        play_audio_flag: Whether to play the audio immediately

    Returns:
        WAV audio data as bytes, or None if error

    """
    # Synthesize speech
    audio_data = await synthesize_speech(
        text=text,
        tts_server_ip=tts_server_ip,
        tts_server_port=tts_server_port,
        logger=logger,
        voice_name=voice_name,
        language=language,
        speaker=speaker,
        console=console,
    )

    # Play audio if requested and synthesis succeeded
    if audio_data and play_audio_flag:
        await play_audio(
            audio_data,
            logger,
            output_device_index=output_device_index,
            console=console,
        )

    return audio_data
