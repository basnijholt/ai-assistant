"""Wyoming TTS Client for converting text to speech."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

import typer
from rich.console import Console

import agent_cli.agents._cli_options as opts
from agent_cli import process_manager
from agent_cli.agents._tts_common import (
    handle_device_listing,
    handle_tts_playback,
    setup_output_device,
)
from agent_cli.audio import pyaudio_context
from agent_cli.cli import app, setup_logging
from agent_cli.utils import (
    get_clipboard_text,
    print_input_panel,
    print_status_message,
)

LOGGER = logging.getLogger()


async def async_main(
    *,
    # General
    quiet: bool,
    console: Console | None,
    # Text input
    text: str | None,
    # TTS parameters
    tts_server_ip: str,
    tts_server_port: int,
    voice_name: str | None,
    tts_language: str | None,
    speaker: str | None,
    # Output device
    output_device_index: int | None,
    output_device_name: str | None,
    list_output_devices_flag: bool,
    # Output file
    save_file: str | None,
) -> None:
    """Async entry point for the speak command."""
    with pyaudio_context() as p:
        # Handle device listing
        if handle_device_listing(p, console, list_output_devices_flag=list_output_devices_flag):
            return

        # Setup output device
        output_device_index, output_device_name = setup_output_device(
            p,
            console,
            output_device_name,
            output_device_index,
        )

        # Get text from argument or clipboard
        if text is None:
            text = get_clipboard_text(console)
            if not text:
                return
            if not quiet and console:
                print_input_panel(console, text, title="📋 Text from Clipboard")
        elif not quiet and console:
            print_input_panel(console, text, title="📝 Text to Speak")

        # Handle TTS playback and saving
        await handle_tts_playback(
            text,
            tts_server_ip=tts_server_ip,
            tts_server_port=tts_server_port,
            voice_name=voice_name,
            tts_language=tts_language,
            speaker=speaker,
            output_device_index=output_device_index,
            save_file=save_file,
            console=console,
            logger=LOGGER,
            play_audio=not save_file,  # Don't play if saving to file
            status_message="🔊 Synthesizing speech..." if console else "",
            description="Audio",
        )


@app.command("speak")
def speak(
    text: str | None = None,
    *,
    # TTS parameters
    tts_server_ip: str = opts.TTS_SERVER_IP,
    tts_server_port: int = opts.TTS_SERVER_PORT,
    voice_name: str | None = opts.VOICE_NAME,
    tts_language: str | None = opts.TTS_LANGUAGE,
    speaker: str | None = opts.SPEAKER,
    # Output device
    output_device_index: int | None = opts.OUTPUT_DEVICE_INDEX,
    output_device_name: str | None = opts.OUTPUT_DEVICE_NAME,
    list_output_devices_flag: bool = opts.LIST_OUTPUT_DEVICES,
    # Output file
    save_file: str | None = typer.Option(
        None,
        "--save-file",
        help="Save audio to WAV file instead of playing it.",
    ),
    # Process control
    stop: bool = opts.STOP,
    status: bool = opts.STATUS,
    # General
    log_level: str = opts.LOG_LEVEL,
    log_file: str | None = opts.LOG_FILE,
    quiet: bool = opts.QUIET,
) -> None:
    """Convert text to speech using Wyoming TTS server.

    If no text is provided, reads from clipboard.

    Usage:
    - Speak text: agent-cli speak "Hello world"
    - Speak from clipboard: agent-cli speak
    - Save to file: agent-cli speak "Hello" --save-file hello.wav
    - Use specific voice: agent-cli speak "Hello" --voice en_US-lessac-medium
    - Run in background: agent-cli speak "Hello" &
    """
    setup_logging(log_level, log_file, quiet=quiet)
    console = Console() if not quiet else None
    process_name = "speak"

    if stop:
        if process_manager.kill_process(process_name):
            print_status_message(console, "✅ Speak stopped.")
        else:
            print_status_message(console, "⚠️  No speak process is running.", style="yellow")
        return

    if status:
        if process_manager.is_process_running(process_name):
            pid = process_manager.read_pid_file(process_name)
            print_status_message(console, f"✅ Speak is running (PID: {pid}).")
        else:
            print_status_message(console, "⚠️  Speak is not running.", style="yellow")
        return

    # Use context manager for PID file management
    with process_manager.pid_file_context(process_name), suppress(KeyboardInterrupt):
        asyncio.run(
            async_main(
                # General
                quiet=quiet,
                console=console,
                # Text input
                text=text,
                # TTS parameters
                tts_server_ip=tts_server_ip,
                tts_server_port=tts_server_port,
                voice_name=voice_name,
                tts_language=tts_language,
                speaker=speaker,
                # Output device
                output_device_index=output_device_index,
                output_device_name=output_device_name,
                list_output_devices_flag=list_output_devices_flag,
                # Output file
                save_file=save_file,
            ),
        )
