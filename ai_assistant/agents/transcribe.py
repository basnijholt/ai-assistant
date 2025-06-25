"""Wyoming ASR Client for streaming microphone audio to a transcription server."""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import suppress
from typing import TYPE_CHECKING

import pyperclip
import typer
from rich.console import Console
from rich.live import Live
from rich.text import Text
from wyoming.client import AsyncClient

from ai_assistant import asr, config, process_manager
from ai_assistant.cli import app, setup_logging
from ai_assistant.utils import _print

if TYPE_CHECKING:
    import pyaudio


async def run_transcription(
    device_index: int | None,
    asr_server_ip: str,
    asr_server_port: int,
    *,
    clipboard: bool,
    quiet: bool,
    p: pyaudio.PyAudio,
) -> None:
    """Connect to the Wyoming server and run the transcription loop."""
    logger = logging.getLogger()
    console = Console() if not quiet else None
    uri = f"tcp://{asr_server_ip}:{asr_server_port}"
    logger.info("Connecting to Wyoming server at %s", uri)

    stop_event = asyncio.Event()

    def shutdown_handler() -> None:
        logger.info("Shutdown signal received.")
        stop_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, shutdown_handler)

    try:
        async with AsyncClient.from_uri(uri) as client:
            logger.info("Connection established")
            _print(console, "[green]Listening...")

            with asr.open_pyaudio_stream(
                p,
                format=config.PYAUDIO_FORMAT,
                channels=config.PYAUDIO_CHANNELS,
                rate=config.PYAUDIO_RATE,
                input=True,
                frames_per_buffer=config.PYAUDIO_CHUNK_SIZE,
                input_device_index=device_index,
            ) as stream:
                live_cm = (
                    Live(
                        Text("Transcribing...", style="blue"),
                        console=console,
                        transient=True,
                    )
                    if console
                    else suppress(Exception)
                )
                with live_cm as live:
                    send_task = asyncio.create_task(
                        asr.send_audio(client, stream, stop_event, logger, live),
                    )
                    recv_task = asyncio.create_task(asr.receive_text(client, logger))
                    await asyncio.gather(send_task, recv_task)

                    transcript = recv_task.result()
                    logger.info("Received transcript: %s", transcript)
                    if transcript and clipboard:
                        pyperclip.copy(transcript)
                        logger.info("Copied transcript to clipboard.")
                        _print(console, "[italic green]Copied to clipboard.[/italic green]")
                    else:
                        logger.info("Transcript empty or clipboard copy disabled.")

    except ConnectionRefusedError:
        _print(
            console,
            f"[bold red]Connection refused.[/bold red] Could not connect to {uri}",
        )
    except Exception:
        logger.exception("Unhandled exception.")
        if console:
            console.print_exception(show_locals=True)


async def async_main(
    device_index: int | None,
    asr_server_ip: str,
    asr_server_port: int,
    *,
    clipboard: bool,
    quiet: bool,
    list_devices: bool,
) -> None:
    """Async entry point, consuming parsed args."""
    console = Console() if not quiet else None
    with asr.pyaudio_context() as p:
        if list_devices:
            asr.list_input_devices(p, console)
            return
        await run_transcription(
            device_index=device_index,
            asr_server_ip=asr_server_ip,
            asr_server_port=asr_server_port,
            clipboard=clipboard,
            quiet=quiet,
            p=p,
        )


@app.command("transcribe")
def transcribe(
    *,
    device_index: int | None = typer.Option(
        None,
        "--device-index",
        help="Index of the PyAudio input device to use.",
    ),
    list_devices: bool = typer.Option(
        False,  # noqa: FBT003
        "--list-devices",
        help="List available audio input devices and exit.",
        is_eager=True,
    ),
    asr_server_ip: str = typer.Option(
        config.ASR_SERVER_IP,
        "--asr-server-ip",
        help="Wyoming ASR server IP address.",
    ),
    asr_server_port: int = typer.Option(
        config.ASR_SERVER_PORT,
        "--asr-server-port",
        help="Wyoming ASR server port.",
    ),
    clipboard: bool = typer.Option(
        True,  # noqa: FBT003
        "--clipboard/--no-clipboard",
        help="Copy transcript to clipboard.",
    ),
    daemon: bool = typer.Option(
        False,  # noqa: FBT003
        "--daemon",
        help="Run as a background daemon process.",
    ),
    kill: bool = typer.Option(
        False,  # noqa: FBT003
        "--kill",
        help="Kill any running transcribe daemon.",
    ),
    status: bool = typer.Option(
        False,  # noqa: FBT003
        "--status",
        help="Check if transcribe daemon is running.",
    ),
    log_level: str = typer.Option(
        "WARNING",
        "--log-level",
        help="Set logging level.",
        case_sensitive=False,
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="Path to a file to write logs to.",
    ),
    quiet: bool = typer.Option(
        False,  # noqa: FBT003
        "-q",
        "--quiet",
        help="Suppress console output from rich.",
    ),
) -> None:
    """Wyoming ASR Client for streaming microphone audio to a transcription server."""
    setup_logging(log_level, log_file, quiet=quiet)
    console = Console() if not quiet else None
    process_name = "transcribe"

    if kill:
        if process_manager.kill_process(process_name):
            _print(console, "[green]✅ Transcribe daemon stopped.[/green]")
        else:
            _print(console, "[yellow]⚠️  No transcribe daemon is running.[/yellow]")
        return

    if status:
        if process_manager.is_process_running(process_name):
            pid = process_manager.read_pid_file(process_name)
            _print(console, f"[green]✅ Transcribe daemon is running (PID: {pid}).[/green]")
        else:
            _print(console, "[yellow]⚠️  Transcribe daemon is not running.[/yellow]")
        return

    def job_to_run() -> None:
        with suppress(KeyboardInterrupt):
            asyncio.run(
                async_main(
                    device_index=device_index,
                    asr_server_ip=asr_server_ip,
                    asr_server_port=asr_server_port,
                    clipboard=clipboard,
                    quiet=quiet,
                    list_devices=list_devices,
                ),
            )

    if daemon:
        process_manager.daemonize(process_name, job_to_run)
    else:
        job_to_run()
