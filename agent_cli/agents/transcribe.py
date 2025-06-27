"""Wyoming ASR Client for streaming microphone audio to a transcription server."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AbstractContextManager, nullcontext, suppress

import pyperclip
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

import agent_cli.agents._cli_options as opts
from agent_cli import asr, process_manager
from agent_cli.cli import app, setup_logging
from agent_cli.llm import process_and_update_clipboard
from agent_cli.utils import _print, print_device_index, signal_handling_context

SYSTEM_PROMPT = """
You are an AI transcription cleanup assistant. Your purpose is to improve and refine raw speech-to-text transcriptions by correcting errors, adding proper punctuation, and enhancing readability while preserving the original meaning and intent.

Your tasks include:
- Correcting obvious speech recognition errors and mishearing
- Adding appropriate punctuation (periods, commas, question marks, etc.)
- Fixing capitalization where needed
- Removing filler words, false starts, and repeated words when they clearly weren't intentional
- Improving sentence structure and flow while maintaining the speaker's voice and meaning
- Formatting the text for better readability

Important rules:
- Do not change the core meaning or content of the transcription
- Do not add information that wasn't spoken
- Do not remove content unless it's clearly an error or filler
- Return ONLY the cleaned-up text without any explanations or commentary
- Do not wrap your output in markdown or code blocks
"""

AGENT_INSTRUCTIONS = """
You will be given a block of raw transcribed text enclosed in <original-text> tags, and a cleanup instruction enclosed in <instruction> tags.

Your job is to process the transcribed text according to the instruction, which will typically involve:
- Correcting speech recognition errors
- Adding proper punctuation and capitalization
- Removing obvious filler words and false starts
- Improving readability while preserving meaning

Return ONLY the cleaned-up text with no additional formatting or commentary.
"""

INSTRUCTION = """
Please clean up this transcribed text by correcting any speech recognition errors, adding appropriate punctuation and capitalization, removing obvious filler words or false starts, and improving overall readability while preserving the original meaning and intent of the speaker.
"""


async def async_main(
    *,
    device_index: int | None,
    device_name: str | None,
    asr_server_ip: str,
    asr_server_port: int,
    clipboard: bool,
    quiet: bool,
    list_devices: bool,
    model: str,
    ollama_host: str,
    llm: bool,
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

        if llm and model and ollama_host and transcript:
            _print(console, Panel(transcript, title="[cyan]Raw Transcript 📝[/cyan]"))
            await process_and_update_clipboard(
                system_prompt=SYSTEM_PROMPT,
                agent_instructions=AGENT_INSTRUCTIONS,
                model=model,
                ollama_host=ollama_host,
                logger=logger,
                console=console,
                original_text=transcript,
                instruction=INSTRUCTION,
                clipboard=clipboard,
            )
            return

        if transcript and clipboard:
            pyperclip.copy(transcript)
            logger.info("Copied transcript to clipboard.")
            _print(console, "[italic blue]Copied to clipboard.[/italic blue]")
        elif not transcript:
            logger.info("Transcript empty.")
        else:
            logger.info("Clipboard copy disabled.")
        if not quiet:
            _print(console, f"[italic green]Transcript: {transcript}[/italic green]")


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
    # ASR
    list_devices: bool = opts.LIST_DEVICES,
    asr_server_ip: str = opts.ASR_SERVER_IP,
    asr_server_port: int = opts.ASR_SERVER_PORT,
    # LLM
    model: str = opts.MODEL,
    ollama_host: str = opts.OLLAMA_HOST,
    llm: bool = opts.LLM,
    # Process control
    stop: bool = opts.STOP,
    status: bool = opts.STATUS,
    # General
    clipboard: bool = opts.CLIPBOARD,
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
                model=model,
                ollama_host=ollama_host,
                llm=llm,
            ),
        )
