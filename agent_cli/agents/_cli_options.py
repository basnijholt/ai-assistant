"""Shared Typer options for the Agent CLI agents."""

from __future__ import annotations

import typer

from agent_cli import config

# --- LLM Options ---
MODEL: str = typer.Option(
    config.DEFAULT_MODEL,
    "--model",
    "-m",
    help=f"The Ollama model to use. Default is {config.DEFAULT_MODEL}.",
)
OLLAMA_HOST: str = typer.Option(
    config.OLLAMA_HOST,
    "--ollama-host",
    help=f"The Ollama server host. Default is {config.OLLAMA_HOST}.",
)


# --- ASR (Audio) Options ---
DEVICE_INDEX: int | None = typer.Option(
    None,
    "--device-index",
    help="Index of the PyAudio input device to use.",
)
DEVICE_NAME: str | None = typer.Option(
    None,
    "--device-name",
    help="Device name keywords for partial matching. Supports comma-separated list where each term can partially match device names (case-insensitive). First matching device is selected.",
)
LIST_DEVICES: bool = typer.Option(
    False,  # noqa: FBT003
    "--list-devices",
    help="List available audio input devices and exit.",
    is_eager=True,
)
ASR_SERVER_IP: str = typer.Option(
    config.ASR_SERVER_IP,
    "--asr-server-ip",
    help="Wyoming ASR server IP address.",
)
ASR_SERVER_PORT: int = typer.Option(
    config.ASR_SERVER_PORT,
    "--asr-server-port",
    help="Wyoming ASR server port.",
)
CLIPBOARD: bool = typer.Option(
    True,  # noqa: FBT003
    "--clipboard/--no-clipboard",
    help="Copy transcript to clipboard.",
)


# --- Process Management Options ---
STOP: bool = typer.Option(
    False,  # noqa: FBT003
    "--stop",
    help="Stop any running background process.",
)
STATUS: bool = typer.Option(
    False,  # noqa: FBT003
    "--status",
    help="Check if a background process is running.",
)


# --- General Options ---
LOG_LEVEL: str = typer.Option(
    "WARNING",
    "--log-level",
    help="Set logging level.",
    case_sensitive=False,
)
LOG_FILE: str | None = typer.Option(
    None,
    "--log-file",
    help="Path to a file to write logs to.",
)
QUIET: bool = typer.Option(
    False,  # noqa: FBT003
    "-q",
    "--quiet",
    help="Suppress console output from rich.",
)
