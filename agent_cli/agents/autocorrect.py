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
from typing import TYPE_CHECKING

import httpx
import pyperclip
import typer
from openai import APIConnectionError
from pydantic_ai.exceptions import ModelHTTPError
from rich.status import Status

import agent_cli.agents._cli_options as opts
from agent_cli.agents._config import GeneralConfig, LLMConfig
from agent_cli.cli import app, setup_logging
from agent_cli.llm import build_agent
from agent_cli.utils import (
    get_clipboard_text,
    print_error_message,
    print_input_panel,
    print_output_panel,
    print_status_message,
)

if TYPE_CHECKING:
    from rich.console import Console

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
    print_input_panel(console, original_text, title="📋 Original Text")


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
        if original_text and corrected_text.strip() == original_text.strip():
            print("✅ No correction needed.")
        else:
            print(corrected_text)
    else:
        assert console is not None
        print_output_panel(
            console,
            corrected_text,
            title="✨ Corrected Text",
            subtitle=f"[dim]took {elapsed:.2f}s[/dim]",
        )
        print_status_message(
            console,
            "✅ Success! Corrected text has been copied to your clipboard.",
        )


async def async_autocorrect(
    *,
    text: str | None,
    llm_config: LLMConfig,
    general_cfg: GeneralConfig,
) -> None:
    """Asynchronous version of the autocorrect command."""
    setup_logging(general_cfg.log_level, general_cfg.log_file, quiet=general_cfg.quiet)
    original_text = text if text is not None else get_clipboard_text(general_cfg.console)

    if original_text is None:
        return

    display_original_text(original_text, general_cfg.console)

    try:
        if general_cfg.quiet:
            corrected_text, elapsed = await process_text(
                original_text,
                llm_config.model,
                llm_config.ollama_host,
            )
        else:
            with Status(
                f"[bold yellow]🤖 Correcting with {llm_config.model}...[/bold yellow]",
                console=general_cfg.console,
            ) as status:
                maybe_log = (
                    f" (see [dim]log at {general_cfg.log_file}[/dim])"
                    if general_cfg.log_file
                    else ""
                )
                status.update(
                    f"[bold yellow]🤖 Correcting with {llm_config.model}...{maybe_log}[/bold yellow]",
                )
                corrected_text, elapsed = await process_text(
                    original_text,
                    llm_config.model,
                    llm_config.ollama_host,
                )

        _display_result(
            corrected_text,
            original_text,
            elapsed,
            simple_output=general_cfg.quiet,
            console=general_cfg.console,
        )

    except (httpx.ConnectError, ModelHTTPError, APIConnectionError) as e:
        if general_cfg.quiet:
            print(f"❌ {e}")
        else:
            print_error_message(
                general_cfg.console,
                str(e),
                f"Please check that your Ollama server is running at [bold cyan]{llm_config.ollama_host}[/bold cyan]",
            )
        sys.exit(1)


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
    llm_config = LLMConfig(model=model, ollama_host=ollama_host)
    general_cfg = GeneralConfig(log_level=log_level, log_file=log_file, quiet=quiet)
    asyncio.run(
        async_autocorrect(
            text=text,
            llm_config=llm_config,
            general_cfg=general_cfg,
        ),
    )
