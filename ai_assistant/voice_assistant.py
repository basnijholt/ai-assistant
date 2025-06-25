r"""Interact with clipboard text via a voice command using Wyoming and an Ollama LLM.

This script combines functionalities from transcribe.py and fix_my_text_ollama.py.

WORKFLOW:
1. The script starts and immediately copies the current content of the clipboard.
2. It then starts listening for a voice command via the microphone.
3. The user triggers a stop signal (e.g., via a Keyboard Maestro hotkey sending SIGINT).
4. The script stops recording and finalizes the transcription of the voice command.
5. It sends the original clipboard text and the transcribed command to a local LLM.
6. The LLM processes the text based on the instruction (either editing it or answering a question).
7. The resulting text is then copied back to the clipboard.

KEYBOARD MAESTRO INTEGRATION:
To create a hotkey toggle for this script, set up a Keyboard Maestro macro with:

1. Trigger: Hot Key (e.g., Cmd+Shift+A for "Assistant")

2. If/Then/Else Action:
   - Condition: Shell script returns success
   - Script: pgrep -f "voice_clipboard_assistant\.py" > /dev/null

3. Then Actions (if process is running):
   - Display Text Briefly: "🗣️ Processing command..."
   - Execute Shell Script: pkill -INT -f "voice_clipboard_assistant\.py"
   - (The script will show its own "Done" notification)

4. Else Actions (if process is not running):
   - Display Text Briefly: "📋 Listening for command..."
   - Execute Shell Script:
     #!/bin/zsh
     source "$HOME/.dotbins/shell/zsh.sh" 2>/dev/null || true  # Adds uv to PATH
     ${HOME}/dotfiles/scripts/voice_clipboard_assistant.py --device-index 1 --quiet &
   - Select "Display results in a notification"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager, nullcontext, suppress
from typing import TYPE_CHECKING, Any

import pyaudio
import pyperclip
from pydantic_ai import Agent
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.status import Status
from rich.text import Text
from wyoming.asr import (
    Transcribe,
    Transcript,
    TranscriptChunk,
    TranscriptStart,
    TranscriptStop,
)
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient

from . import cli
from .ollama_client import build_agent
from .utils import get_clipboard_text

# --- Configuration ---
ASR_SERVER_IP = "192.168.1.143"
ASR_SERVER_PORT = 10300
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "devstral:24b"

# PyAudio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 1024

# LLM Prompts
SYSTEM_PROMPT = """\
You are a versatile AI text assistant. Your purpose is to either **modify** a given text or **answer questions** about it, based on a specific instruction.

- If the instruction is a **command to edit** the text (e.g., "make this more formal," "add emojis," "correct spelling"), you must return ONLY the full, modified text.
- If the instruction is a **question about** the text (e.g., "summarize this," "what are the key points?," "translate to French"), you must return ONLY the answer.

In all cases, you must follow these strict rules:
- Do not provide any explanations, apologies, or introductory phrases like "Here is the result:".
- Do not wrap your output in markdown or code blocks.
- Your output should be the direct result of the instruction: either the edited text or the answer to the question.
"""

AGENT_INSTRUCTIONS = """\
You will be given a block of text enclosed in <original-text> tags, and an instruction enclosed in <instruction> tags.
Analyze the instruction to determine if it's a command to edit the text or a question about it.

- If it is an editing command, apply the changes to the original text and return the complete, modified version.
- If it is a question, formulate an answer based on the original text.

Return ONLY the resulting text (either the edit or the answer), with no extra formatting or commentary.
"""

if TYPE_CHECKING:
    from collections.abc import Generator

    from pydantic_ai import Agent
    from rich.align import Align


# --- Helper Functions & Context Managers ---


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interact with clipboard text with a voice command using Wyoming and an Ollama LLM.",
    )
    # Audio arguments
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
        default=ASR_SERVER_IP,
        help="Wyoming ASR server IP address.",
    )
    parser.add_argument(
        "--asr-server-port",
        type=int,
        default=ASR_SERVER_PORT,
        help="Wyoming ASR server port.",
    )
    # LLM arguments
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"The Ollama model to use. Default is {DEFAULT_MODEL}.",
    )
    # General arguments
    parser.add_argument("--log-file", help="Path to log file (default: stdout only).")
    parser.add_argument(
        "--log-level",
        type=str.upper,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: WARNING).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Don't print anything to the console.",
    )
    return parser.parse_args()


def setup_logging(args: argparse.Namespace) -> logging.Logger:
    """Set up logging to console and optionally a file."""
    handlers = [logging.StreamHandler()] if not args.quiet else []
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="w"))  # type: ignore[arg-type]
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def _print(console: Console | None, message: str | Align, **kwargs: Any) -> None:
    if console is not None:
        console.print(message, **kwargs)


@contextmanager
def pyaudio_context() -> Generator[pyaudio.PyAudio, None, None]:
    """Context manager for PyAudio lifecycle."""
    p = pyaudio.PyAudio()
    try:
        yield p
    finally:
        p.terminate()


@contextmanager
def open_pyaudio_stream(
    p: pyaudio.PyAudio,
    *args,
    **kwargs,
) -> Generator[pyaudio.Stream, None, None]:
    """Context manager for a PyAudio stream that ensures it's properly closed."""
    stream = p.open(*args, **kwargs)
    try:
        yield stream
    finally:
        stream.stop_stream()
        stream.close()


def list_input_devices(pa: pyaudio.PyAudio, console: Console | None) -> None:
    """Print a numbered list of available input devices."""
    _print(console, "[bold]Available input devices:[/bold]")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            _print(console, f"  [yellow]{i}[/yellow]: {info['name']}")


# --- ASR (Transcription) Logic ---


async def send_audio(
    client: AsyncClient,
    stream: pyaudio.Stream,
    stop_event: asyncio.Event,
    logger: logging.Logger,
    console: Console | None,
) -> None:
    """Read from mic and send to Wyoming server."""
    await client.write_event(Transcribe().event())
    await client.write_event(AudioStart(rate=RATE, width=2, channels=CHANNELS).event())

    try:
        live_cm = (
            Live(
                Text("Listening...", style="blue"),
                console=console,
                transient=True,
                refresh_per_second=10,
            )
            if console
            else nullcontext()
        )
        with live_cm as live:
            seconds_streamed = 0.0
            while not stop_event.is_set():
                chunk = await asyncio.to_thread(
                    stream.read,
                    num_frames=CHUNK_SIZE,
                    exception_on_overflow=False,
                )
                await client.write_event(
                    AudioChunk(
                        rate=RATE,
                        width=2,
                        channels=CHANNELS,
                        audio=chunk,
                    ).event(),
                )
                logger.debug("Sent %d byte(s) of audio", len(chunk))
                if console:
                    seconds_streamed += len(chunk) / (RATE * CHANNELS * 2)
                    assert isinstance(live, Live)
                    live.update(
                        Text(f"Listening... ({seconds_streamed:.1f}s)", style="blue"),
                    )
    finally:
        await client.write_event(AudioStop().event())
        logger.debug("Sent AudioStop")


async def receive_text(
    client: AsyncClient,
    logger: logging.Logger,
    console: Console | None,
) -> str:
    """Receive transcription events and return the final transcript."""
    transcript_text = ""
    while True:
        event = await client.read_event()
        if event is None:
            logger.warning("Connection to ASR server lost.")
            break

        if Transcript.is_type(event.type):
            transcript = Transcript.from_event(event)
            transcript_text = transcript.text
            _print(
                console,
                f"\n[bold green]Instruction:[/bold green] {transcript_text}",
            )
            logger.info("Final transcript: %s", transcript_text)
            break
        if TranscriptChunk.is_type(event.type):
            chunk = TranscriptChunk.from_event(event)
            _print(console, chunk.text, end="")
            logger.debug("Transcript chunk: %s", chunk.text)
        elif TranscriptStart.is_type(event.type) or TranscriptStop.is_type(event.type):
            logger.debug("Received %s", event.type)
        else:
            logger.debug("Ignoring event type: %s", event.type)

    return transcript_text


async def get_voice_instruction(
    args: argparse.Namespace,
    logger: logging.Logger,
    p: pyaudio.PyAudio,
    stop_event: asyncio.Event,
    console: Console | None,
) -> str | None:
    """Connects to ASR server and returns the transcribed instruction."""
    uri = f"tcp://{args.asr_server_ip}:{args.asr_server_port}"
    logger.info("Connecting to Wyoming server at %s", uri)

    try:
        async with AsyncClient.from_uri(uri) as client:
            logger.info("ASR connection established")
            _print(console, "[green]Listening for your command...[/green]")

            with open_pyaudio_stream(
                p,
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                input_device_index=args.device_index,
            ) as stream:
                send_task = asyncio.create_task(
                    send_audio(client, stream, stop_event, logger, console),
                )
                recv_task = asyncio.create_task(receive_text(client, logger, console))
                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.ALL_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                # The result of recv_task is the transcript string
                return next(t.result() for t in done if t is recv_task)
    except ConnectionRefusedError:
        _print(
            console,
            f"[bold red]ASR Connection refused.[/bold red] Is the server at {uri} running?",
        )
        return None
    except Exception as e:
        logger.exception("An error occurred during transcription.")
        _print(console, f"[bold red]Transcription error:[/bold red] {e}")
        return None


# --- LLM (Editing) Logic ---


async def process_with_llm(
    agent: Agent,
    original_text: str,
    instruction: str,
) -> tuple[str, float]:
    """Run the agent asynchronously and return corrected text and elapsed time."""
    t_start = time.monotonic()
    user_input = f"""
<original-text>
{original_text}
</original-text>

<instruction>
{instruction}
</instruction>
"""
    result = await agent.run(user_input)
    t_end = time.monotonic()
    return result.output, t_end - t_start


async def process_and_update_clipboard(
    args: argparse.Namespace,
    logger: logging.Logger,
    console: Console | None,
    original_text: str,
    instruction: str,
) -> None:
    """Processes the text with the LLM, updates the clipboard, and displays the result.

    In quiet mode, only the result is printed to stdout.
    """
    agent = build_agent(
        model=args.model,
        ollama_host=OLLAMA_HOST,
        system_prompt=SYSTEM_PROMPT,
        instructions=AGENT_INSTRUCTIONS,
    )
    try:
        status_cm = (
            Status(
                f"[bold yellow]🤖 Applying instruction with {args.model}...[/bold yellow]",
                console=console,
            )
            if console
            else nullcontext()
        )
        with status_cm:
            result_text, elapsed = await process_with_llm(
                agent,
                original_text,
                instruction,
            )

        pyperclip.copy(result_text)
        logger.info("Copied result to clipboard.")

        if console:
            console.print(
                Panel(
                    result_text,
                    title="[bold green]✨ Result (Copied to Clipboard)[/bold green]",
                    border_style="green",
                    subtitle=f"[dim]took {elapsed:.2f}s[/dim]",
                ),
            )
        else:
            # Quiet mode: print result to stdout for Keyboard Maestro to capture
            print(result_text)

    except Exception as e:
        logger.exception("An error occurred during LLM processing.")
        _print(
            console,
            f"❌ [bold red]An unexpected LLM error occurred: {e}[/bold red]",
        )
        _print(
            console,
            f"   Please check your Ollama server at [cyan]{OLLAMA_HOST}[/cyan]",
        )
        sys.exit(1)


# --- Main Application Logic ---


async def main() -> None:
    """Main function."""
    parser = cli.get_base_parser()
    parser.description = __doc__
    # Add script-specific arguments
    # ...
    args = parser.parse_args()
    cli.setup_logging(args)
    logger = logging.getLogger()

    console = Console() if not args.quiet else None

    with pyaudio_context() as p:
        if args.list_devices:
            list_input_devices(p, console)
            return

        original_text = get_clipboard_text(console)
        if not original_text:
            return

        _print(console, Panel(original_text, title="[cyan]📝 Text to Process[/cyan]"))
        if args.device_index is not None:
            _print(
                console,
                f"🎤 Using device [bold yellow]{args.device_index}[/bold yellow]",
            )
        else:
            _print(
                console,
                "[bold yellow]⚠️  No --device-index specified. Using default system input.[/bold yellow]",
            )

        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def shutdown_handler() -> None:
            logger.info("Shutdown signal received. Stopping transcription.")
            if not stop_event.is_set():
                stop_event.set()

        loop.add_signal_handler(signal.SIGINT, shutdown_handler)
        loop.add_signal_handler(signal.SIGTERM, shutdown_handler)

        instruction = await get_voice_instruction(args, logger, p, stop_event, console)

        if not instruction or not instruction.strip():
            _print(console, "[yellow]No instruction was transcribed. Exiting.[/yellow]")
            return

        await process_and_update_clipboard(
            args,
            logger,
            console,
            original_text,
            instruction,
        )


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        asyncio.run(main())
