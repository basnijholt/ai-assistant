"""Wyoming ASR Client for streaming microphone audio to a transcription server."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AbstractContextManager, nullcontext, suppress

import pyperclip
from rich.console import Console
from rich.live import Live
from rich.text import Text

import agent_cli.agents._cli_options as opts
from agent_cli import asr, process_manager
from agent_cli.cli import app, setup_logging
from agent_cli.utils import _print, print_device_index, signal_handling_context


async def async_main(
    device_index: int | None,
    device_name: str | None,
    asr_server_ip: str,
    asr_server_port: int,
    *,
    clipboard: bool,
    quiet: bool,
    list_devices: bool,
) -> None:
    """Async entry point, consuming parsed args."""
    logger = logging.getLogger()
    console = Console() if not quiet else None

    with asr.pyaudio_context() as p:
        if list_devices:
            asr.list_input_devices(p, console)
            return
        device_index, device_name = asr.input_device(p, device_name, device_index)
        print_device_index(console, device_index, device_name)

        with (
            signal_handling_context(console, logger) as stop_event,
            _maybe_live(console) as live,
        ):
            transcript = await asr.transcribe_audio(
                asr_server_ip=asr_server_ip,
                asr_server_port=asr_server_port,
                device_index=device_index,
                logger=logger,
                p=p,
                stop_event=stop_event,
                console=console,
                live=live,
                listening_message="Listening...",
            )

            if transcript and clipboard:
                pyperclip.copy(transcript)
                logger.info("Copied transcript to clipboard.")
                _print(console, "[italic green]Copied to clipboard.[/italic green]")
            elif not transcript:
                logger.info("Transcript empty.")
            else:
                logger.info("Clipboard copy disabled.")


def _maybe_live(console: Console | None) -> AbstractContextManager[Live | None]:
    if console:
        return Live(
            Text("Transcribing...", style="blue"),
            console=console,
            transient=True,
        )
    return nullcontext()


@app.command("transcribe")
def transcribe(
    *,
    device_index: int | None = opts.DEVICE_INDEX,
    device_name: str | None = opts.DEVICE_NAME,
    list_devices: bool = opts.LIST_DEVICES,
    asr_server_ip: str = opts.ASR_SERVER_IP,
    asr_server_port: int = opts.ASR_SERVER_PORT,
    clipboard: bool = opts.CLIPBOARD,
    stop: bool = opts.STOP,
    status: bool = opts.STATUS,
    log_level: str = opts.LOG_LEVEL,
    log_file: str | None = opts.LOG_FILE,
    quiet: bool = opts.QUIET,
) -> None:
    """Wyoming ASR Client for streaming microphone audio to a transcription server.

    Usage:
    - Run in foreground: agent-cli transcribe --device-index 1
    - Run in background: agent-cli transcribe --device-index 1 &
    - Check status: agent-cli transcribe --status
    - Stop background process: agent-cli transcribe --stop
    """
    setup_logging(log_level, log_file, quiet=quiet)
    console = Console() if not quiet else None
    process_name = "transcribe"

    if stop:
        if process_manager.kill_process(process_name):
            _print(console, "[green]✅ Transcribe stopped.[/green]")
        else:
            _print(console, "[yellow]⚠️  No transcribe is running.[/yellow]")
        return

    if status:
        if process_manager.is_process_running(process_name):
            pid = process_manager.read_pid_file(process_name)
            _print(console, f"[green]✅ Transcribe is running (PID: {pid}).[/green]")
        else:
            _print(console, "[yellow]⚠️  Transcribe is not running.[/yellow]")
        return

    # Use context manager for PID file management
    with process_manager.pid_file_context(process_name), suppress(KeyboardInterrupt):
        asyncio.run(
            async_main(
                device_index=device_index,
                device_name=device_name,
                asr_server_ip=asr_server_ip,
                asr_server_port=asr_server_port,
                clipboard=clipboard,
                quiet=quiet,
                list_devices=list_devices,
            ),
        )
