"""Utility functions for agent CLI operations."""

from __future__ import annotations

import asyncio
import signal
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol, TypeVar

import pyperclip
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.status import Status
from rich.text import Text

from agent_cli import process_manager

if TYPE_CHECKING:
    import logging
    from collections.abc import Generator
    from datetime import timedelta

console = Console()

T = TypeVar("T")


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
        import asyncio

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


def create_spinner(text: str) -> Spinner:
    """Creates a default spinner."""
    return Spinner("dots", text=Text(text, style="blue"))


def create_status(text: str) -> Status:
    """Creates a default status."""
    return Status(create_spinner(text), console=console)


def print_input_panel(
    text: str,
    title: str = "Input",
    style: str = "bold blue",
) -> None:
    """Prints a panel with the input text."""
    console.print(Panel(text, title=title, border_style=style))


def print_output_panel(
    text: str,
    title: str = "Output",
    subtitle: str = "",
    style: str = "bold green",
) -> None:
    """Prints a panel with the output text."""
    console.print(Panel(text, title=title, subtitle=subtitle, border_style=style))


def print_error_message(message: str, suggestion: str | None = None) -> None:
    """Prints an error message in a panel."""
    error_text = Text(message)
    if suggestion:
        error_text.append("\n\n")
        error_text.append(suggestion)
    console.print(Panel(error_text, title="Error", border_style="bold red"))


def print_status_message(message: str, style: str = "bold green") -> None:
    """Prints a status message."""
    console.print(f"[{style}]{message}[/{style}]")


def print_device_index(device_index: int | None, device_name: str | None) -> None:
    """Prints the device index."""
    if device_index is not None:
        name = device_name or "Unknown Device"
        print_status_message(f"Using {name} device with index {device_index}")


def get_clipboard_text(*, quiet: bool = False) -> str | None:
    """Get text from clipboard, with an optional status message."""
    text = pyperclip.paste()
    if not text:
        if not quiet:
            print_status_message("Clipboard is empty.", style="yellow")
        return None
    return text


@contextmanager
def signal_handling_context(
    logger: logging.Logger,
    quiet: bool = False,
) -> Generator[InteractiveStopEvent, None, None]:
    """Context manager for graceful signal handling with double Ctrl+C support.

    Sets up handlers for SIGINT (Ctrl+C) and SIGTERM (kill command):
    - First Ctrl+C: Graceful shutdown with warning message
    - Second Ctrl+C: Force exit with code 130
    - SIGTERM: Immediate graceful shutdown

    Args:
        logger: Logger instance for recording events
        quiet: Whether to suppress console output

    Yields:
        stop_event: InteractiveStopEvent that gets set when shutdown is requested

    """
    stop_event = InteractiveStopEvent()

    def sigint_handler() -> None:
        sigint_count = stop_event.increment_sigint_count()

        if sigint_count == 1:
            logger.info("First Ctrl+C received. Processing transcription.")
            if not quiet:
                console.print(
                    "\n[yellow]Ctrl+C pressed. Processing transcription... (Press Ctrl+C again to force exit)[/yellow]",
                )
            stop_event.set()
        else:
            logger.info("Second Ctrl+C received. Force exiting.")
            if not quiet:
                console.print("\n[red]Force exit![/red]")
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


def stop_or_status(
    process_name: str,
    which: str,
    stop: bool,
    status: bool,
    *,
    quiet: bool = False,
) -> bool:
    """Handle process control for a given process name."""
    if stop:
        if process_manager.kill_process(process_name):
            if not quiet:
                print_status_message(f"✅ {which.capitalize()} stopped.")
        elif not quiet:
            print_status_message(f"⚠️  No {which} is running.", style="yellow")
        return True

    if status:
        if process_manager.is_process_running(process_name):
            pid = process_manager.read_pid_file(process_name)
            if not quiet:
                print_status_message(f"✅ {which.capitalize()} is running (PID: {pid}).")
        elif not quiet:
            print_status_message(f"⚠️ {which.capitalize()} is not running.", style="yellow")
        return True

    return False
