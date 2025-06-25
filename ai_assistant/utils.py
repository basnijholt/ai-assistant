"""Common utility functions for the AI assistant tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyperclip

if TYPE_CHECKING:
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
