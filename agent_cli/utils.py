"""Common utility functions for the Agent CLI tools."""

from __future__ import annotations

import asyncio
import signal
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

import pyperclip
from rich.panel import Panel

from agent_cli import process_manager

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


class Stoppable(Protocol):
    """Protocol for objects that can be stopped, like asyncio.Event."""

    def is_set(self) -> bool:
        """Return true if the event is set."""
        ...

    def set(self) -> None:
        """Set the event."""
        ...

    def clear(self) -> None:
        """Clear the event."""
        ...


class InteractiveStopEvent:
    """A stop event with reset capability for interactive agents."""

    def __init__(self) -> None:
        """Initialize the interactive stop event."""
        self._event = asyncio.Event()
        self._sigint_count = 0

    def is_set(self) -> bool:
        """Check if the stop event is set."""
        return self._event.is_set()

    def set(self) -> None:
        """Set the stop event."""
        self._event.set()

    def clear(self) -> None:
        """Clear the stop event and reset interrupt count for next iteration."""
        self._event.clear()
        self._sigint_count = 0

    def increment_sigint_count(self) -> int:
        """Increment and return the SIGINT count."""
        self._sigint_count += 1
        return self._sigint_count


@contextmanager
def signal_handling_context(
    console: Console | None,
    logger: logging.Logger,
) -> Generator[InteractiveStopEvent, None, None]:
    """Context manager for graceful signal handling with double Ctrl+C support.

    Sets up handlers for SIGINT (Ctrl+C) and SIGTERM (kill command):
    - First Ctrl+C: Graceful shutdown with warning message
    - Second Ctrl+C: Force exit with code 130
    - SIGTERM: Immediate graceful shutdown

    Args:
        console: Rich console for user messages (None for quiet mode)
        logger: Logger instance for recording events

    Yields:
        stop_event: InteractiveStopEvent that gets set when shutdown is requested

    """
    stop_event = InteractiveStopEvent()

    def sigint_handler() -> None:
        sigint_count = stop_event.increment_sigint_count()

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


def stop_or_status(
    process_name: str,
    which: str,
    console: Console,
    stop: bool,  # noqa: FBT001
    status: bool,  # noqa: FBT001
) -> bool:
    """Handle process control for a given process name."""
    if stop:
        if process_manager.kill_process(process_name):
            print_status_message(console, f"✅ {which.capitalize()} stopped.")
        else:
            print_status_message(console, f"⚠️  No {which} is running.", style="yellow")
        return True

    if status:
        if process_manager.is_process_running(process_name):
            pid = process_manager.read_pid_file(process_name)
            print_status_message(console, f"✅ {which.capitalize()} is running (PID: {pid}).")
        else:
            print_status_message(console, f"⚠️ {which.capitalize()} is not running.", style="yellow")
        return True

    return False
