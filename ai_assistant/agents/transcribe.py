"""Wyoming ASR Client for streaming microphone audio to a transcription server."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from contextlib import nullcontext, suppress
from typing import TYPE_CHECKING

import pyperclip
from rich.console import Console
from rich.live import Live
from rich.text import Text

from ai_assistant import asr, cli
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
        async with asr.AsyncClient.from_uri(uri) as client:
            logger.info("Connection established")
            _print(console, "[green]Listening...")

            with asr.open_pyaudio_stream(
                p,
                format=asr.FORMAT,
                channels=asr.CHANNELS,
                rate=asr.RATE,
                input=True,
                frames_per_buffer=asr.CHUNK_SIZE,
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
                    if transcript and args.clipboard:
                        pyperclip.copy(transcript)
                        logger.info("Copied transcript to clipboard.")
                        _print(console, "[italic green]Copied to clipboard.[/italic green]")

    except ConnectionRefusedError:
        _print(
            console,
            f"[bold red]Connection refused.[/bold red] Could not connect to {uri}",
        )
    except Exception:
        logger.exception("Unhandled exception.")
        if console:
            console.print_exception(show_locals=True)


async def async_main() -> None:
    """Main entry point."""
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
        default="192.168.1.143",
        help="Wyoming ASR server IP address.",
    )
    parser.add_argument(
        "--asr-server-port",
        type=int,
        default=10300,
        help="Wyoming ASR server port.",
    )
    parser.add_argument(
        "--clipboard",
        action="store_true",
        default=True,
        help="Copy transcript to clipboard (default: True).",
    )

    args = parser.parse_args()
    cli.setup_logging(args)

    logger = logging.getLogger()
    console = Console() if not args.quiet else None

    with asr.pyaudio_context() as p:
        if args.list_devices:
            asr.list_input_devices(p, console)
            sys.exit(0)
        await run_transcription(args, logger, p, console)


def main() -> None:
    """Synchronous entry point for CLI."""
    with suppress(KeyboardInterrupt):
        asyncio.run(async_main())


if __name__ == "__main__":
    main()
