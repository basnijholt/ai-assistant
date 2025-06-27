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
   - Execute Shell Script: voice-assistant --stop --quiet
   - (The script will show its own "Done" notification)

4. Else Actions (if process is not running):
   - Display Text Briefly: "📋 Listening for command..."
   - Execute Shell Script: voice-assistant --device-index 1 --quiet &
   - Select "Display results in a notification"

This approach uses standard Unix background processes (&) instead of Python daemons!
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextlib import AbstractContextManager, nullcontext, suppress
from typing import TYPE_CHECKING

import pyperclip
from rich.console import Console
from rich.panel import Panel
from rich.status import Status

import agent_cli.agents._cli_options as opts
from agent_cli import asr, process_manager
from agent_cli.cli import app, setup_logging
from agent_cli.ollama_client import build_agent
from agent_cli.utils import (
    _print,
    get_clipboard_text,
    print_device_index,
    signal_handling_context,
)

if TYPE_CHECKING:
    from pydantic_ai import Agent


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


# --- LLM (Editing) Logic ---

INPUT_TEMPLATE = """
<original-text>
{original_text}
</original-text>

<instruction>
{instruction}
</instruction>
"""


async def process_with_llm(
    agent: Agent,
    original_text: str,
    instruction: str,
) -> tuple[str, float]:
    """Run the agent asynchronously and return corrected text and elapsed time."""
    user_input = INPUT_TEMPLATE.format(original_text=original_text, instruction=instruction)
    t_start = time.monotonic()
    result = await agent.run(user_input)
    t_end = time.monotonic()
    return result.output, t_end - t_start


def _maybe_status(console: Console | None, model: str) -> AbstractContextManager[Status | None]:
    """Context manager for status display."""
    if console:
        return Status(
            f"[bold yellow]🤖 Applying instruction with {model}...[/bold yellow]",
            console=console,
        )
    return nullcontext()


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
    agent = build_agent(
        model=model,
        ollama_host=ollama_host,
        system_prompt=SYSTEM_PROMPT,
        instructions=AGENT_INSTRUCTIONS,
    )
    try:
        with _maybe_status(console, model):
            result_text, elapsed = await process_with_llm(agent, original_text, instruction)

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
    device_name: str | None,
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

        device_index, device_name = asr.input_device(p, device_name, device_index)

        original_text = get_clipboard_text(console)
        if not original_text:
            return

        print_device_index(console, device_index, device_name)
        _print(console, Panel(original_text, title="[cyan]📝 Text to Process[/cyan]"))

        with signal_handling_context(console, logger) as stop_event:
            # Define callbacks for voice assistant specific formatting
            def chunk_callback(chunk_text: str) -> None:
                """Handle transcript chunks as they arrive."""
                _print(console, chunk_text, end="")

            def final_callback(transcript_text: str) -> None:
                """Format the final instruction result."""
                _print(
                    console,
                    f"\n[bold green]Instruction:[/bold green] {transcript_text}",
                )

            instruction = await asr.transcribe_audio(
                asr_server_ip=asr_server_ip,
                asr_server_port=asr_server_port,
                device_index=device_index,
                logger=logger,
                p=p,
                stop_event=stop_event,
                console=console,
                listening_message="Listening for your command...",
                chunk_callback=chunk_callback,
                final_callback=final_callback,
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
    device_name: str | None = opts.DEVICE_NAME,
    *,
    list_devices: bool = opts.LIST_DEVICES,
    asr_server_ip: str = opts.ASR_SERVER_IP,
    asr_server_port: int = opts.ASR_SERVER_PORT,
    model: str = opts.MODEL,
    ollama_host: str = opts.OLLAMA_HOST,
    stop: bool = opts.STOP,
    status: bool = opts.STATUS,
    log_level: str = opts.LOG_LEVEL,
    log_file: str | None = opts.LOG_FILE,
    quiet: bool = opts.QUIET,
) -> None:
    """Interact with clipboard text via a voice command using Wyoming and an Ollama LLM.

    Usage:
    - Run in foreground: agent-cli voice-assistant --device-index 1
    - Run in background: agent-cli voice-assistant --device-index 1 &
    - Check status: agent-cli voice-assistant --status
    - Stop background process: agent-cli voice-assistant --stop
    """
    setup_logging(log_level, log_file, quiet=quiet)
    console = Console() if not quiet else None
    process_name = "voice-assistant"

    if stop:
        if process_manager.kill_process(process_name):
            _print(console, "[green]✅ Voice assistant stopped.[/green]")
        else:
            _print(console, "[yellow]⚠️  No voice assistant is running.[/yellow]")
        return

    if status:
        if process_manager.is_process_running(process_name):
            pid = process_manager.read_pid_file(process_name)
            _print(
                console,
                f"[green]✅ Voice assistant is running (PID: {pid}).[/green]",
            )
        else:
            _print(console, "[yellow]⚠️  Voice assistant is not running.[/yellow]")
        return

    # Use context manager for PID file management
    with process_manager.pid_file_context(process_name), suppress(KeyboardInterrupt):
        asyncio.run(
            async_main(
                quiet=quiet,
                device_index=device_index,
                device_name=device_name,
                list_devices=list_devices,
                asr_server_ip=asr_server_ip,
                asr_server_port=asr_server_port,
                model=model,
                ollama_host=ollama_host,
            ),
        )
