"""Wyoming TTS Client for converting text to speech."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path  # noqa: TC003

import typer

import agent_cli.agents._cli_options as opts
from agent_cli import process_manager
from agent_cli.agents._config import FileConfig, GeneralConfig, TTSConfig
from agent_cli.agents._tts_common import handle_tts_playback
from agent_cli.audio import list_output_devices, output_device, pyaudio_context
from agent_cli.cli import app, setup_logging
from agent_cli.utils import (
    get_clipboard_text,
    print_input_panel,
    print_status_message,
    stop_or_status,
)

LOGGER = logging.getLogger()


async def async_main(
    *,
    general_cfg: GeneralConfig,
    text: str | None,
    tts_config: TTSConfig,
    file_config: FileConfig,
) -> None:
    """Async entry point for the speak command."""
    with pyaudio_context() as p:
        # Handle device listing
        if tts_config.list_output_devices:
            list_output_devices(p, general_cfg.console)
            return

        # Setup output device
        output_device_index, output_device_name = output_device(
            p,
            tts_config.output_device_name,
            tts_config.output_device_index,
        )
        if output_device_index is not None and general_cfg.console:
            msg = f"🔊 Using output device [bold yellow]{output_device_index}[/bold yellow] ([italic]{output_device_name}[/italic])"
            print_status_message(general_cfg.console, msg)

        # Get text from argument or clipboard
        if text is None:
            text = get_clipboard_text(general_cfg.console)
            if not text:
                return
            if not general_cfg.quiet and general_cfg.console:
                print_input_panel(general_cfg.console, text, title="📋 Text from Clipboard")
        elif not general_cfg.quiet and general_cfg.console:
            print_input_panel(general_cfg.console, text, title="📝 Text to Speak")

        # Handle TTS playback and saving
        await handle_tts_playback(
            text,
            tts_server_ip=tts_config.server_ip,
            tts_server_port=tts_config.server_port,
            voice_name=tts_config.voice_name,
            tts_language=tts_config.language,
            speaker=tts_config.speaker,
            output_device_index=output_device_index,
            save_file=file_config.save_file,
            console=general_cfg.console,
            logger=LOGGER,
            play_audio=not file_config.save_file,  # Don't play if saving to file
            status_message="🔊 Synthesizing speech..." if general_cfg.console else "",
            description="Audio",
        )


@app.command("speak")
def speak(
    text: str | None = typer.Argument(
        None,
        help="Text to speak. Reads from clipboard if not provided.",
    ),
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
    save_file: Path | None = typer.Option(  # noqa: B008
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
    general_cfg = GeneralConfig(log_level=log_level, log_file=log_file, quiet=quiet)
    process_name = "speak"
    if stop_or_status(process_name, "speak process", general_cfg.console, stop, status):
        return

    # Use context manager for PID file management
    with process_manager.pid_file_context(process_name), suppress(KeyboardInterrupt):
        tts_config = TTSConfig(
            enabled=True,  # Implied for speak command
            server_ip=tts_server_ip,
            server_port=tts_server_port,
            voice_name=voice_name,
            language=tts_language,
            speaker=speaker,
            output_device_index=output_device_index,
            output_device_name=output_device_name,
            list_output_devices=list_output_devices_flag,
        )
        file_config = FileConfig(save_file=save_file)

        asyncio.run(
            async_main(
                general_cfg=general_cfg,
                text=text,
                tts_config=tts_config,
                file_config=file_config,
            ),
        )
