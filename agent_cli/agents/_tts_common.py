"""Shared TTS utilities for speak and voice-assistant commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from agent_cli import tts
from agent_cli.utils import Stoppable, print_status_message

if TYPE_CHECKING:
    import logging

    from rich.console import Console


async def _save_audio_file(
    audio_data: bytes,
    save_file: Path,
    console: Console | None,
    logger: logging.Logger,
    *,
    description: str = "Audio",
) -> None:
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
    save_file: Path | None,
    console: Console | None,
    logger: logging.Logger,
    play_audio: bool = True,
    status_message: str = "🔊 Speaking...",
    description: str = "Audio",
    stop_event: Stoppable | None = None,
    speed: float = 1.0,
) -> bytes | None:
    """Handle TTS synthesis, playback, and file saving."""
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
            stop_event=stop_event,
            speed=speed,
        )

        if save_file and audio_data:
            await _save_audio_file(
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
