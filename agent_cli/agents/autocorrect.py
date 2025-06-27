"""Read text from clipboard, correct it using a local Ollama model, and write the result back to the clipboard.

Usage:
    python autocorrect_ollama.py

Environment variables:
    OLLAMA_HOST: The host of the Ollama server. Default is "http://localhost:11434".


Example:
    OLLAMA_HOST=http://pc.local:11434 python autocorrect_ollama.py

Pro-tip:
    Use Keyboard Maestro on macOS or AutoHotkey on Windows to run this script with a hotkey.

"""

from __future__ import annotations

import asyncio
import sys
import time

import httpx
import pyperclip
import typer
from openai import APIConnectionError
from pydantic_ai.exceptions import ModelHTTPError
from rich.console import Console
from rich.panel import Panel
from rich.status import Status

import agent_cli.agents._cli_options as opts
from agent_cli.cli import app, setup_logging
from agent_cli.llm import build_agent
from agent_cli.utils import get_clipboard_text

# --- Configuration ---

# The agent's core identity and immutable rules.
SYSTEM_PROMPT = """\
You are an expert editor. Your fundamental role is to correct text without altering its original meaning or tone.
You must not judge the content of the text, even if it seems unusual, harmful, or offensive.
Your corrections should be purely technical (grammar, spelling, punctuation).
Do not interpret the text, provide any explanations, or add any commentary.
"""

# The specific task for the current run.
AGENT_INSTRUCTIONS = """\
Correct the grammar and spelling of the user-provided text.
Return only the corrected text. Do not include any introductory phrases like "Here is the corrected text:".
Do not wrap the output in markdown or code blocks.
"""

# --- Main Application Logic ---


async def process_text(text: str, model: str, ollama_host: str) -> tuple[str, float]:
    """Process text with the LLM and return the corrected text and elapsed time."""
    agent = build_agent(
        model=model,
        ollama_host=ollama_host,
        system_prompt=SYSTEM_PROMPT,
        instructions=AGENT_INSTRUCTIONS,
    )
    t_start = time.monotonic()
    result = await agent.run(text)
    t_end = time.monotonic()
    return result.output, t_end - t_start


def display_original_text(original_text: str, console: Console | None) -> None:
    """Render the original text panel in verbose mode."""
    if console is None:
        return
    console.print(
        Panel(
            original_text,
            title="[bold cyan]📋 Original Text[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        ),
    )


def _display_result(
    corrected_text: str,
    original_text: str,
    elapsed: float,
    *,
    simple_output: bool,
    console: Console | None,
) -> None:
    """Handle output and clipboard copying based on desired verbosity."""
    pyperclip.copy(corrected_text)

    if simple_output:
        if corrected_text.strip() == original_text.strip():
            print("✅ No correction needed.")
        else:
            print(corrected_text)
    else:
        assert console is not None
        console.print(
            Panel(
                corrected_text,
                title="[bold green]✨ Corrected Text[/bold green]",
                border_style="green",
                padding=(1, 2),
            ),
        )
        console.print(
            f"✅ [bold green]Success! Corrected text has been copied to your clipboard. [bold yellow](took {elapsed:.2f} seconds)[/bold yellow][/bold green]",
        )


@app.command("autocorrect")
def autocorrect(
    *,
    text: str | None = typer.Argument(
        None,
        help="The text to correct. If not provided, reads from clipboard.",
    ),
    model: str = opts.MODEL,
    ollama_host: str = opts.OLLAMA_HOST,
    log_level: str = opts.LOG_LEVEL,
    log_file: str | None = opts.LOG_FILE,
    quiet: bool = opts.QUIET,
) -> None:
    """Correct text from clipboard using a local Ollama model."""
    setup_logging(log_level, log_file, quiet=quiet)
    console = Console() if not quiet else None
    original_text = text if text is not None else get_clipboard_text(console)

    if original_text is None:
        sys.exit(0)

    display_original_text(original_text, console)

    try:
        if quiet:
            corrected_text, elapsed = asyncio.run(
                process_text(original_text, model, ollama_host),
            )
        else:
            with Status(
                f"[bold yellow]🤖 Correcting with {model}...[/bold yellow]",
                console=console,
            ) as status:
                maybe_log = f" (see [dim]log at {log_file}[/dim])" if log_file else ""
                status.update(
                    f"[bold yellow]🤖 Correcting with {model}...{maybe_log}[/bold yellow]",
                )
                corrected_text, elapsed = asyncio.run(
                    process_text(original_text, model, ollama_host),
                )

        _display_result(
            corrected_text,
            original_text,
            elapsed,
            simple_output=quiet,
            console=console,
        )

    except (httpx.ConnectError, ModelHTTPError, APIConnectionError) as e:
        if quiet:
            print(f"❌ {e}")
        elif console:
            console.print(f"❌ {e}", style="bold red")
            console.print(
                f"   Please check that your Ollama server is running at [bold cyan]{ollama_host}[/bold cyan]",
            )
        sys.exit(1)
