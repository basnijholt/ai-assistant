"""Wyoming TTS Client for converting text to speech."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

import typer
from rich.console import Console

import agent_cli.agents._cli_options as opts
from agent_cli import process_manager, tts
from agent_cli.asr import list_output_devices, output_device, pyaudio_context  # Reuse ASR utilities
from agent_cli.cli import app, setup_logging
from agent_cli.utils import (
    get_clipboard_text,
    print_input_panel,
    print_status_message,
)

LOGGER = logging.getLogger()


async def async_main(
    *,
    text: str | None,
    tts_server_ip: str,
    tts_server_port: int,
    voice_name: str | None,
    tts_language: str | None,
    speaker: str | None,
    output_device_index: int | None,
    output_device_name: str | None,
    list_output_devices_flag: bool,
    save_file: str | None,
    quiet: bool,
    console: Console | None,
) -> None:
    """Async entry point for the speak command."""
    with pyaudio_context() as p:  # Reuse ASR context manager
        if list_output_devices_flag:
            list_output_devices(p, console)  # Reuse ASR device listing
            return

        # Get output device info
        output_device_index, output_device_name = output_device(
            p,
            output_device_name,
            output_device_index,
        )
        if output_device_index is not None and console:
            # Show output device info (adapted message for TTS)
            msg = f"🔊 Using output device [bold yellow]{output_device_index}[/bold yellow] ([italic]{output_device_name}[/italic])"
            print_status_message(console, msg)

        # Get text from argument or clipboard
        if text is None:
            text = get_clipboard_text(console)
            if not text:
                return
            if not quiet and console:
                print_input_panel(console, text, title="📋 Text from Clipboard")
        elif not quiet and console:
            print_input_panel(console, text, title="📝 Text to Speak")

        # Synthesize and play speech
        audio_data = await tts.speak_text(
            text=text,
            tts_server_ip=tts_server_ip,
            tts_server_port=tts_server_port,
            logger=LOGGER,
            voice_name=voice_name,
            language=tts_language,
            speaker=speaker,
            output_device_index=output_device_index,
            console=console,
            play_audio_flag=not save_file,  # Don't play if saving to file
        )

        # Save to file if requested
        if save_file and audio_data:
            try:
                with open(save_file, "wb") as f:
                    f.write(audio_data)
                if console:
                    print_status_message(console, f"💾 Audio saved to {save_file}")
                LOGGER.info("Audio saved to %s", save_file)
            except Exception as e:
                LOGGER.error("Failed to save audio: %s", e)
                if console:
                    print_status_message(console, f"❌ Failed to save audio: {e}", style="red")


@app.command("speak")
def speak(
    text: str | None = None,
    *,
    # TTS
    tts_server_ip: str = opts.TTS_SERVER_IP,
    tts_server_port: int = opts.TTS_SERVER_PORT,
    voice_name: str | None = opts.VOICE_NAME,
    tts_language: str | None = opts.TTS_LANGUAGE,
    speaker: str | None = opts.SPEAKER,
    # Output device
    output_device_index: int | None = opts.OUTPUT_DEVICE_INDEX,
    output_device_name: str | None = opts.OUTPUT_DEVICE_NAME,
    list_output_devices_flag: bool = opts.LIST_OUTPUT_DEVICES,
    # Output
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
                text=text,
                tts_server_ip=tts_server_ip,
                tts_server_port=tts_server_port,
                voice_name=voice_name,
                tts_language=tts_language,
                speaker=speaker,
                output_device_index=output_device_index,
                output_device_name=output_device_name,
                list_output_devices_flag=list_output_devices_flag,
                save_file=save_file,
                quiet=quiet,
                console=console,
            ),
        )
