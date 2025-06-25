"""Read text from clipboard, correct it using a local Ollama model, and write the result back to the clipboard.

Usage:
    python fix_my_text_ollama.py

Environment variables:
    OLLAMA_HOST: The host of the Ollama server. Default is "http://localhost:11434".


Example:
    OLLAMA_HOST=http://pc.local:11434 python fix_my_text_ollama.py

Pro-tip:
    Use Keyboard Maestro on macOS or AutoHotkey on Windows to run this script with a hotkey.

"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx
import pyperclip
from ollama._types import OllamaError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console
from rich.panel import Panel
from rich.status import Status

# --- Configuration ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "gemma3:latest"

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return the parsed namespace."""
    parser = argparse.ArgumentParser(
        description="Correct text from clipboard using a local Ollama model.",
    )
    parser.add_argument(
        "--simple-output",
        "-s",
        action="store_true",
        help="Print minimal output (suitable for notifications/automation).",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"The Ollama model to use. Default is {DEFAULT_MODEL}.",
    )
    return parser.parse_args()


def build_agent(model: str) -> Agent:
    """Construct and return a PydanticAI agent configured for local Ollama."""
    ollama_provider = OpenAIProvider(base_url=f"{OLLAMA_HOST}/v1")
    ollama_model = OpenAIModel(
        model_name=model,
        provider=ollama_provider,
    )
    return Agent(
        model=ollama_model,
        system_prompt=SYSTEM_PROMPT,
        instructions=AGENT_INSTRUCTIONS,
    )


def process_text(agent: Agent, text: str) -> tuple[str, float]:
    """Run the agent synchronously and return corrected text along with elapsed seconds."""
    t_start = time.monotonic()
    result = agent.run_sync(text)
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


def main() -> None:
    """Orchestrate argument parsing, processing, and output."""
    args = parse_args()
    simple_output = args.simple_output

    console: Console | None = Console() if not simple_output else None

    original_text = pyperclip.paste()

    if not original_text or not original_text.strip():
        message = "❌ Clipboard is empty. Nothing to correct."
        if simple_output:
            print(message)
        else:
            assert console is not None
            console.print(f"[bold red]{message}[/bold red]")
        sys.exit(0)

    display_original_text(original_text, console)

    agent = build_agent(args.model)

    try:
        if simple_output:
            corrected_text, elapsed = process_text(agent, original_text)
        else:
            assert console is not None
            with Status(
                "[bold yellow]🤖 Processing text with Ollama model...[/bold yellow]",
                console=console,
            ):
                corrected_text, elapsed = process_text(agent, original_text)

        _display_result(
            corrected_text,
            original_text,
            elapsed,
            simple_output=simple_output,
            console=console,
        )

    except (OllamaError, httpx.ConnectError) as e:
        if simple_output:
            print(f"❌ {e}")
        else:
            assert console is not None
            console.print(f"❌ [bold red]An unexpected error occurred: {e}[/bold red]")
            console.print(
                f"   Please check that your Ollama server is running at [bold cyan]{OLLAMA_HOST}[/bold cyan]",
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
