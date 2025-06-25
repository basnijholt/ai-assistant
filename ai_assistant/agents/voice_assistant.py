r"""Interact with clipboard text via a voice command using Wyoming and an Ollama LLM.

This script combines functionalities from transcribe.py and autocorrect_ollama.py.

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
   - Script: voice-assistant --status >/dev/null 2>&1

3. Then Actions (if process is running):
   - Display Text Briefly: "🗣️ Processing command..."
   - Execute Shell Script: voice-assistant --kill --quiet
   - (The script will show its own "Done" notification)

4. Else Actions (if process is not running):
   - Display Text Briefly: "📋 Listening for command..."
   - Execute Shell Script: voice-assistant --daemon --device-index 1 --quiet
   - Select "Display results in a notification"

This is much cleaner than the previous approach using pgrep/pkill!
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from contextlib import nullcontext, suppress
from typing import TYPE_CHECKING, Any

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

import ai_assistant.agents._cli_options as opts
from ai_assistant import asr, config, process_manager
from ai_assistant.cli import app, setup_logging
from ai_assistant.ollama_client import build_agent
from ai_assistant.utils import get_clipboard_text

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
    import pyaudio
    from pydantic_ai import Agent
    from rich.align import Align
    from wyoming.client import AsyncClient


# --- Helper Functions & Context Managers ---


def _print(console: Console | None, message: str | Align, **kwargs: Any) -> None:
    if console is not None:
        console.print(message, **kwargs)


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
    await client.write_event(
        AudioStart(
            rate=config.PYAUDIO_RATE,
            width=2,
            channels=config.PYAUDIO_CHANNELS,
        ).event(),
    )

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
                    num_frames=config.PYAUDIO_CHUNK_SIZE,
                    exception_on_overflow=False,
                )
                await client.write_event(
                    AudioChunk(
                        rate=config.PYAUDIO_RATE,
                        width=2,
                        channels=config.PYAUDIO_CHANNELS,
                        audio=chunk,
                    ).event(),
                )
                logger.debug("Sent %d byte(s) of audio", len(chunk))
                if console:
                    seconds_streamed += len(chunk) / (
                        config.PYAUDIO_RATE * config.PYAUDIO_CHANNELS * 2
                    )
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
    asr_server_ip: str,
    asr_server_port: int,
    device_index: int | None,
    logger: logging.Logger,
    p: pyaudio.PyAudio,
    stop_event: asyncio.Event,
    console: Console | None,
) -> str | None:
    """Connects to ASR server and returns the transcribed instruction."""
    uri = f"tcp://{asr_server_ip}:{asr_server_port}"
    logger.info("Connecting to Wyoming server at %s", uri)

    try:
        async with AsyncClient.from_uri(uri) as client:
            logger.info("ASR connection established")
            _print(console, "[green]Listening for your command...[/green]")
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
            with (
                asr.open_pyaudio_stream(
                    p,
                    format=config.PYAUDIO_FORMAT,
                    channels=config.PYAUDIO_CHANNELS,
                    rate=config.PYAUDIO_RATE,
                    input=True,
                    frames_per_buffer=config.PYAUDIO_CHUNK_SIZE,
                    input_device_index=device_index,
                ) as stream,
                live_cm,
            ):
                with live_cm:
                    send_task = asyncio.create_task(
                        send_audio(client, stream, stop_event, logger, console),
                    )
                    recv_task = asyncio.create_task(
                        receive_text(client, logger, console)
                    )
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
    result = await agent.run(
        user_input,
        system_prompt=SYSTEM_PROMPT,
        instructions=AGENT_INSTRUCTIONS,
    )
    t_end = time.monotonic()
    return result.output, t_end - t_start


async def process_and_update_clipboard(
    model: str,
    ollama_host: str,
    logger: logging.Logger,
    console: Console | None,
    original_text: str,
    instruction: str,
) -> None:
    """Processes the text with the LLM, updates the clipboard, and displays the result.

    In quiet mode, only the result is printed to stdout.
    """
    agent = build_agent(model=model, ollama_host=ollama_host)
    try:
        status_cm = (
            Status(
                f"[bold yellow]🤖 Applying instruction with {model}...[/bold yellow]",
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
            f"   Please check your Ollama server at [cyan]{ollama_host}[/cyan]",
        )
        sys.exit(1)


# --- Main Application Logic ---


async def async_main(
    *,
    quiet: bool,
    device_index: int | None,
    list_devices: bool,
    asr_server_ip: str,
    asr_server_port: int,
    model: str,
    ollama_host: str,
) -> None:
    """Main async function, consumes parsed arguments."""
    logger = logging.getLogger()
    console = Console() if not quiet else None

    with asr.pyaudio_context() as p:
        if list_devices:
            asr.list_input_devices(p, console)
            return

        original_text = get_clipboard_text(console)
        if not original_text:
            return

        _print(console, Panel(original_text, title="[cyan]📝 Text to Process[/cyan]"))
        if device_index is not None:
            _print(
                console,
                f"🎤 Using device [bold yellow]{device_index}[/bold yellow]",
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

        instruction = await get_voice_instruction(
            asr_server_ip=asr_server_ip,
            asr_server_port=asr_server_port,
            device_index=device_index,
            logger=logger,
            p=p,
            stop_event=stop_event,
            console=console,
        )

        if not instruction or not instruction.strip():
            _print(console, "[yellow]No instruction was transcribed. Exiting.[/yellow]")
            return

        await process_and_update_clipboard(
            model=model,
            ollama_host=ollama_host,
            logger=logger,
            console=console,
            original_text=original_text,
            instruction=instruction,
        )


@app.command("voice-assistant")
def voice_assistant(
    device_index: int | None = opts.DEVICE_INDEX,
    *,
    list_devices: bool = opts.LIST_DEVICES,
    asr_server_ip: str = opts.ASR_SERVER_IP,
    asr_server_port: int = opts.ASR_SERVER_PORT,
    model: str = opts.MODEL,
    ollama_host: str = opts.OLLAMA_HOST,
    daemon: bool = opts.DAEMON,
    kill: bool = opts.KILL,
    status: bool = opts.STATUS,
    log_level: str = opts.LOG_LEVEL,
    log_file: str | None = opts.LOG_FILE,
    quiet: bool = opts.QUIET,
) -> None:
    """Interact with clipboard text via a voice command using Wyoming and an Ollama LLM."""
    setup_logging(log_level, log_file, quiet=quiet)
    console = Console() if not quiet else None
    process_name = "voice-assistant"

    if kill:
        if process_manager.kill_process(process_name):
            _print(console, "[green]✅ Voice assistant daemon stopped.[/green]")
        else:
            _print(console, "[yellow]⚠️  No voice assistant daemon is running.[/yellow]")
        return

    if status:
        if process_manager.is_process_running(process_name):
            pid = process_manager.read_pid_file(process_name)
            _print(
                console,
                f"[green]✅ Voice assistant daemon is running (PID: {pid}).[/green]",
            )
        else:
            _print(console, "[yellow]⚠️  No voice assistant daemon is running.[/yellow]")
        return

    def job_to_run() -> None:
        with suppress(KeyboardInterrupt):
            asyncio.run(
                async_main(
                    quiet=quiet,
                    device_index=device_index,
                    list_devices=list_devices,
                    asr_server_ip=asr_server_ip,
                    asr_server_port=asr_server_port,
                    model=model,
                    ollama_host=ollama_host,
                ),
            )

    if daemon:
        process_manager.daemonize(process_name, job_to_run)
    else:
        job_to_run()
