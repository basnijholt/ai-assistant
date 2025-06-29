"""Shared CLI functionality for the Agent CLI tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import typer
from rich.console import Console

if TYPE_CHECKING:
    from logging import Handler


app = typer.Typer(
    name="agent-cli",
    help="A suite of AI-powered command-line tools for text correction, audio transcription, and voice assistance.",
    add_completion=True,
)

console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """A suite of AI-powered tools."""
    if ctx.invoked_subcommand is None:
        console.print("[bold red]No command specified.[/bold red]")
        console.print("[bold yellow]Running --help for your convenience.[/bold yellow]")
        console.print(ctx.get_help())


def setup_logging(log_level: str, log_file: str | None, *, quiet: bool) -> None:
    """Sets up logging based on parsed arguments."""
    handlers: list[Handler] = []
    if not quiet:
        handlers.append(logging.StreamHandler())
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w"))

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


# Import commands from other modules to register them
from .agents import autocorrect, interactive, speak, transcribe, voice_assistant  # noqa: E402, F401
