"""Wyoming ASR Client for streaming microphone audio to a transcription server."""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import nullcontext, suppress
from typing import TYPE_CHECKING

import pyperclip
from rich.console import Console
from rich.live import Live
from rich.text import Text
from wyoming.client import AsyncClient

from ai_assistant import asr, cli, config, process_manager
from ai_assistant.utils import _print

if TYPE_CHECKING:
    import argparse

    import pyaudio


async def run_transcription(
    args: argparse.Namespace,
    logger: logging.Logger,
    p: pyaudio.PyAudio,
    console: Console | None,
) -> None:
    """Connect to the Wyoming server and run the transcription loop."""
    uri = f"tcp://{args.asr_server_ip}:{args.asr_server_port}"
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
                input_device_index=args.device_index,
            ) as stream:
                live_cm = (
                    Live(
                        Text("Transcribing...", style="blue"),
                        console=console,
                        transient=True,
                    )
                    if console
                    else nullcontext()
                )
                with live_cm as live:
                    send_task = asyncio.create_task(
                        asr.send_audio(client, stream, stop_event, logger, live),
                    )
                    recv_task = asyncio.create_task(asr.receive_text(client, logger))
                    await asyncio.gather(send_task, recv_task)

                    transcript = recv_task.result()
                    logger.info("Received transcript: %s", transcript)
                    if transcript and args.clipboard:
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


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point, consuming parsed args."""
    logger = logging.getLogger()
    console = Console() if not args.quiet else None

    with asr.pyaudio_context() as p:
        if args.list_devices:
            asr.list_input_devices(p, console)
            return
        await run_transcription(args, logger, p, console)


def main() -> None:
    """Synchronous entry point for CLI."""
    parser = cli.get_base_parser()
    parser.description = __doc__

    # Add transcribe-specific arguments
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Index of the PyAudio input device to use.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit.",
    )
    parser.add_argument(
        "--asr-server-ip",
        default=config.ASR_SERVER_IP,
        help="Wyoming ASR server IP address.",
    )
    parser.add_argument(
        "--asr-server-port",
        type=int,
        default=config.ASR_SERVER_PORT,
        help="Wyoming ASR server port.",
    )
    parser.add_argument(
        "--clipboard",
        action="store_true",
        default=True,
        help="Copy transcript to clipboard (default: True).",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a background daemon process.",
    )
    parser.add_argument(
        "--kill",
        action="store_true",
        help="Kill any running transcribe daemon.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check if transcribe daemon is running.",
    )

    args = parser.parse_args()
    cli.setup_logging(args)
    console = Console() if not args.quiet else None

    process_name = "transcribe"
    if args.kill:
        if process_manager.kill_process(process_name):
            _print(console, "[green]✅ Transcribe daemon stopped.[/green]")
        else:
            _print(console, "[yellow]⚠️  No transcribe daemon is running.[/yellow]")
        return

    if args.status:
        if process_manager.is_process_running(process_name):
            pid = process_manager.read_pid_file(process_name)
            _print(console, f"[green]✅ Transcribe daemon is running (PID: {pid}).[/green]")
        else:
            _print(console, "[yellow]⚠️  Transcribe daemon is not running.[/yellow]")
        return

    def job_to_run() -> None:
        with suppress(KeyboardInterrupt):
            asyncio.run(async_main(args))

    if args.daemon:
        process_manager.daemonize(process_name, job_to_run)
    else:
        job_to_run()


if __name__ == "__main__":
    main()
