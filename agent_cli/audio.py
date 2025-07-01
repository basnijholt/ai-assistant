"""General audio utilities for PyAudio device management and audio streaming."""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pyaudio

from agent_cli.utils import console

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def pyaudio_context() -> Generator[pyaudio.PyAudio, None, None]:
    """Context manager for PyAudio lifecycle."""
    p = pyaudio.PyAudio()
    try:
        yield p
    finally:
        p.terminate()


@contextmanager
def open_pyaudio_stream(
    p: pyaudio.PyAudio,
    *args: object,
    **kwargs: object,
) -> Generator[pyaudio.Stream, None, None]:
    """Context manager for a PyAudio stream that ensures it's properly closed."""
    stream = p.open(*args, **kwargs)
    try:
        yield stream
    finally:
        stream.stop_stream()
        stream.close()


@functools.cache
def get_all_devices(p: pyaudio.PyAudio) -> list[dict]:
    """Get information for all audio devices with caching.

    Args:
        p: PyAudio instance

    Returns:
        List of device info dictionaries with added 'index' field

    """
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        # Add the index to the info dict for convenience
        device_info = dict(info)
        device_info["index"] = i
        devices.append(device_info)
    return devices


def get_device_by_index(p: pyaudio.PyAudio, device_index: int) -> dict:
    """Get device info by index from cached device list.

    Args:
        p: PyAudio instance
        device_index: Device index to look up

    Returns:
        Device info dictionary

    Raises:
        ValueError: If device index is not found

    """
    for device in get_all_devices(p):
        if device["index"] == device_index:
            return device
    msg = f"Device index {device_index} not found"
    raise ValueError(msg)


def list_input_devices(p: pyaudio.PyAudio, quiet: bool = False) -> None:
    """Print a numbered list of available input devices."""
    if not quiet:
        console.print("[bold]Available input devices:[/bold]")
        for device in get_all_devices(p):
            if device.get("maxInputChannels", 0) > 0:
                console.print(f"  [yellow]{device['index']}[/yellow]: {device['name']}")


def list_output_devices(p: pyaudio.PyAudio, quiet: bool = False) -> None:
    """Print a numbered list of available output devices."""
    if not quiet:
        console.print("[bold]Available output devices:[/bold]")
        for device in get_all_devices(p):
            if device.get("maxOutputChannels", 0) > 0:
                console.print(f"  [yellow]{device['index']}[/yellow]: {device['name']}")


def list_all_devices(p: pyaudio.PyAudio, quiet: bool = False) -> None:
    """Print a numbered list of all available audio devices with their capabilities."""
    if not quiet:
        console.print("[bold]All available audio devices:[/bold]")
        for device in get_all_devices(p):
            input_channels = device.get("maxInputChannels", 0)
            output_channels = device.get("maxOutputChannels", 0)

            capabilities = []
            if input_channels > 0:
                capabilities.append(f"{input_channels} input")
            if output_channels > 0:
                capabilities.append(f"{output_channels} output")

            if capabilities:
                cap_str = " (" + ", ".join(capabilities) + ")"
                console.print(f"  [yellow]{device['index']}[/yellow]: {device['name']}{cap_str}")


def _in_or_out_device(
    p: pyaudio.PyAudio,
    device_name: str | None,
    device_index: int | None,
    key: str,
    what: str,
) -> tuple[int | None, str | None]:
    """Find an input device by a prioritized, comma-separated list of keywords."""
    if device_name is None and device_index is None:
        return None, None

    if device_index is not None:
        info = get_device_by_index(p, device_index)
        return device_index, info.get("name")
    assert device_name is not None
    search_terms = [term.strip().lower() for term in device_name.split(",") if term.strip()]

    if not search_terms:
        msg = "Device name string is empty or contains only whitespace."
        raise ValueError(msg)

    devices = []
    for device in get_all_devices(p):
        device_info_name = device.get("name")
        if device_info_name and device.get(key, 0) > 0:
            devices.append((device["index"], device_info_name))

    for term in search_terms:
        for index, name in devices:
            if term in name.lower():
                return index, name

    msg = f"No {what} device found matching any of the keywords in {device_name!r}"
    raise ValueError(msg)


def input_device(
    p: pyaudio.PyAudio,
    device_name: str | None,
    device_index: int | None,
) -> tuple[int | None, str | None]:
    """Find an input device by a prioritized, comma-separated list of keywords."""
    return _in_or_out_device(p, device_name, device_index, "maxInputChannels", "input")


def output_device(
    p: pyaudio.PyAudio,
    device_name: str | None,
    device_index: int | None,
) -> tuple[int | None, str | None]:
    """Find an output device by a prioritized, comma-separated list of keywords."""
    return _in_or_out_device(p, device_name, device_index, "maxOutputChannels", "output")
