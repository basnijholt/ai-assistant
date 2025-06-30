"""Data classes for agent configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import Console


@dataclass
class LLMConfig:
    """LLM configuration parameters."""

    model: str
    ollama_host: str


@dataclass
class ASRConfig:
    """ASR configuration parameters."""

    server_ip: str
    server_port: int
    device_index: int | None
    device_name: str | None
    list_devices: bool


@dataclass
class TTSConfig:
    """TTS configuration parameters."""

    enabled: bool
    server_ip: str
    server_port: int
    voice_name: str | None
    language: str | None
    speaker: str | None
    output_device_index: int | None
    output_device_name: str | None
    list_output_devices: bool


@dataclass
class GeneralConfig:
    """General configuration parameters."""

    log_level: str
    log_file: str | None
    quiet: bool
    console: Console | None
    clipboard: bool = True  # Default value since not all agents have it


@dataclass
class FileConfig:
    """File-related configuration."""

    save_file: Path | None
    history_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.history_dir:
            self.history_dir = self.history_dir.expanduser()
        if self.save_file:
            self.save_file = self.save_file.expanduser()
