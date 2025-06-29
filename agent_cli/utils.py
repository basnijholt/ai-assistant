"""Common utility functions for the Agent CLI tools."""

from __future__ import annotations

import asyncio
import signal
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pyperclip
from rich.panel import Panel

if TYPE_CHECKING:
    import logging
    from collections.abc import Generator
    from datetime import timedelta

    from rich.align import Align
    from rich.console import Console


def _print(console: Console | None, message: str | Align, **kwargs: object) -> None:
    if console:
        console.print(message, **kwargs)


def get_clipboard_text(console: Console | None) -> str | None:
    """Retrieves text from the clipboard."""
    try:
        original_text = pyperclip.paste()
        if not original_text or not original_text.strip():
            _print(console, "❌ Clipboard is empty.")
            return None
        return original_text
    except pyperclip.PyperclipException:
        _print(console, "❌ Could not read from clipboard.")
        return None


@contextmanager
def signal_handling_context(
    console: Console | None,
    logger: logging.Logger,
) -> Generator[asyncio.Event, None, None]:
    """Context manager for graceful signal handling with double Ctrl+C support.

    Sets up handlers for SIGINT (Ctrl+C) and SIGTERM (kill command):
    - First Ctrl+C: Graceful shutdown with warning message
    - Second Ctrl+C: Force exit with code 130
    - SIGTERM: Immediate graceful shutdown

    Args:
        console: Rich console for user messages (None for quiet mode)
        logger: Logger instance for recording events

    Yields:
        stop_event: asyncio.Event that gets set when shutdown is requested

    """
    stop_event = asyncio.Event()
    sigint_count = 0

    def sigint_handler() -> None:
        nonlocal sigint_count
        sigint_count += 1

        if sigint_count == 1:
            logger.info("First Ctrl+C received. Processing transcription.")
            _print(
                console,
                "\n[yellow]Ctrl+C pressed. Processing transcription... (Press Ctrl+C again to force exit)[/yellow]",
            )
            stop_event.set()
        else:
            logger.info("Second Ctrl+C received. Force exiting.")
            _print(console, "\n[red]Force exit![/red]")
            sys.exit(130)  # Standard exit code for Ctrl+C

    def sigterm_handler() -> None:
        logger.info("SIGTERM received. Stopping process.")
        stop_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, sigint_handler)
    loop.add_signal_handler(signal.SIGTERM, sigterm_handler)

    try:
        yield stop_event
    finally:
        # Signal handlers are automatically cleaned up when the event loop exits
        pass


def print_device_index(
    console: Console | None,
    device_index: int | None,
    device_name: str | None,
) -> None:
    """Print the device index to the console."""
    if device_index is not None:
        msg = f"🎤 Using device [bold yellow]{device_index}[/bold yellow] ([italic]{device_name}[/italic])"
    else:
        msg = (
            "[bold yellow]⚠️  No --device-index specified. Using default system input.[/bold yellow]"
        )
    _print(console, msg)


# Standardized output formatting functions
def print_input_panel(
    console: Console | None,
    text: str,
    title: str = "📋 Input",
    border_style: str = "cyan",
) -> None:
    """Print input text in a standardized panel format."""
    if console:
        console.print(
            Panel(
                text,
                title=f"[bold {border_style}]{title}[/bold {border_style}]",
                border_style=border_style,
                padding=(1, 2),
            ),
        )


def print_output_panel(
    console: Console | None,
    text: str,
    title: str = "✨ Output",
    border_style: str = "green",
    subtitle: str | None = None,
) -> None:
    """Print output text in a standardized panel format."""
    if console:
        console.print(
            Panel(
                text,
                title=f"[bold {border_style}]{title}[/bold {border_style}]",
                border_style=border_style,
                subtitle=subtitle,
                padding=(1, 2),
            ),
        )


def print_status_message(
    console: Console | None,
    message: str,
    style: str = "green",
) -> None:
    """Print a status message with consistent formatting."""
    if console:
        console.print(f"[{style}]{message}[/{style}]")


def print_error_message(
    console: Console | None,
    message: str,
    detail: str | None = None,
) -> None:
    """Print an error message with consistent formatting."""
    if console:
        console.print(f"[bold red]❌ {message}[/bold red]")
        if detail:
            console.print(f"   {detail}")


def format_timedelta_to_ago(td: timedelta) -> str:
    """Format a timedelta into a human-readable 'ago' string."""
    seconds = int(td.total_seconds())
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    if days > 0:
        return f"{days} day{'s' if days != 1 else ''} ago"
    if hours > 0:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    if minutes > 0:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    return f"{seconds} second{'s' if seconds != 1 else ''} ago"
