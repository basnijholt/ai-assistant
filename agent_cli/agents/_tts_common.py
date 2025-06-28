"""Shared TTS utilities for speak and voice-assistant commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from agent_cli import tts
from agent_cli.audio import list_output_devices, output_device
from agent_cli.utils import print_status_message

if TYPE_CHECKING:
    import logging

    import pyaudio
    from rich.console import Console


def setup_output_device(
    p: pyaudio.PyAudio,
    console: Console | None,
    output_device_name: str | None,
    output_device_index: int | None,
) -> tuple[int | None, str | None]:
    """Setup and display output device info.

    Args:
        p: PyAudio instance
        console: Rich console for output
        output_device_name: Device name keywords for matching
        output_device_index: Specific device index

    Returns:
        Tuple of (device_index, device_name)

    """
    device_index, device_name = output_device(
        p,
        output_device_name,
        output_device_index,
    )

    if device_index is not None and console:
        msg = f"🔊 Using output device [bold yellow]{device_index}[/bold yellow] ([italic]{device_name}[/italic])"
        print_status_message(console, msg)

    return device_index, device_name


async def save_audio_file(
    audio_data: bytes,
    save_file: str,
    console: Console | None,
    logger: logging.Logger,
    *,
    description: str = "Audio",
) -> None:
    """Save audio data to a file with error handling.

    Args:
        audio_data: Audio data to save
        save_file: File path to save to
        console: Rich console for output
        logger: Logger instance
        description: Description for log messages (e.g., "Audio", "TTS audio")

    """
    try:
        save_path = Path(save_file)
        await asyncio.to_thread(save_path.write_bytes, audio_data)
        if console:
            print_status_message(console, f"💾 {description} saved to {save_file}")
        logger.info("%s saved to %s", description, save_file)
    except (OSError, PermissionError) as e:
        logger.exception("Failed to save %s", description.lower())
        if console:
            print_status_message(
                console,
                f"❌ Failed to save {description.lower()}: {e}",
                style="red",
            )


async def handle_tts_playback(
    text: str,
    *,
    tts_server_ip: str,
    tts_server_port: int,
    voice_name: str | None,
    tts_language: str | None,
    speaker: str | None,
    output_device_index: int | None,
    save_file: str | None,
    console: Console | None,
    logger: logging.Logger,
    play_audio: bool = True,
    status_message: str = "🔊 Speaking...",
    description: str = "Audio",
) -> bytes | None:
    """Handle TTS synthesis, playback, and file saving.

    Args:
        text: Text to synthesize
        tts_server_ip: Wyoming TTS server IP
        tts_server_port: Wyoming TTS server port
        voice_name: Optional voice name
        tts_language: Optional language
        speaker: Optional speaker name
        output_device_index: Optional output device index
        save_file: Optional file path to save audio
        console: Rich console for output
        logger: Logger instance
        play_audio: Whether to play audio immediately
        status_message: Message to display during synthesis
        description: Description for save file logging

    Returns:
        Audio data bytes if successful, None otherwise

    """
    try:
        if console and status_message:
            print_status_message(console, status_message, style="blue")

        audio_data = await tts.speak_text(
            text=text,
            tts_server_ip=tts_server_ip,
            tts_server_port=tts_server_port,
            logger=logger,
            voice_name=voice_name,
            language=tts_language,
            speaker=speaker,
            output_device_index=output_device_index,
            console=console,
            play_audio_flag=play_audio,
        )

        # Save audio to file if requested
        if save_file and audio_data:
            await save_audio_file(
                audio_data,
                save_file,
                console,
                logger,
                description=description,
            )

        return audio_data

    except (OSError, ConnectionError, TimeoutError) as e:
        logger.warning("Failed TTS operation: %s", e)
        if console:
            print_status_message(console, f"⚠️ TTS failed: {e}", style="yellow")
        return None


def handle_device_listing(
    p: pyaudio.PyAudio,
    console: Console | None,
    *,
    list_output_devices_flag: bool,
) -> bool:
    """Handle device listing commands.

    Args:
        p: PyAudio instance
        console: Rich console for output
        list_output_devices_flag: Whether to list output devices

    Returns:
        True if a listing was performed (caller should return early)

    """
    if list_output_devices_flag:
        list_output_devices(p, console)
        return True
    return False
