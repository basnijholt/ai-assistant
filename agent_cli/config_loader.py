"""Handles loading and parsing of the agent-cli configuration file."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .utils import console

CONFIG_PATH = Path.home() / ".config/agent-cli/config.toml"


def load_config(config_path_str: str | None = None) -> dict[str, Any]:
    """Load the TOML configuration file."""
    config_path = Path(config_path_str) if config_path_str else CONFIG_PATH

    if config_path.exists():
        with config_path.open("rb") as f:
            return tomllib.load(f)
    elif config_path_str:
        # If a specific path was given and not found, it's an error
        console.print(
            f"[bold red]Config file not found at {config_path_str}[/bold red]",
        )
    return {}
