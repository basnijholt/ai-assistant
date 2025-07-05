"""General audio utilities for PyAudio device management and audio streaming."""

from __future__ import annotations

import asyncio
import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pyaudio
from rich.text import Text

from agent_cli.utils import InteractiveStopEvent, console

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable, Generator

    from rich.live import Live


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


async def read_audio_stream(
    stream: pyaudio.Stream,
    stop_event: InteractiveStopEvent,
    chunk_handler: Callable[[bytes], None] | Callable[[bytes], asyncio.Awaitable[None]],
    logger: logging.Logger,
    *,
    live: Live | None = None,
    quiet: bool = False,
    progress_message: str = "Processing audio",
    progress_style: str = "blue",
) -> None:
    """Core audio reading function - reads chunks and calls handler.

    This is the single source of truth for audio reading logic.
    All other audio functions should use this to avoid duplication.

    Args:
        stream: PyAudio stream
        stop_event: Event to stop reading
        chunk_handler: Function to handle each chunk (sync or async)
        logger: Logger instance
        live: Rich Live display for progress
        quiet: If True, suppress console output
        progress_message: Message to display
        progress_style: Rich style for progress

    """
    from agent_cli import config

    try:
        seconds_streamed = 0.0
        while not stop_event.is_set():
            chunk = await asyncio.to_thread(
                stream.read,
                num_frames=config.PYAUDIO_CHUNK_SIZE,
                exception_on_overflow=False,
            )

            # Handle chunk (sync or async)
            if asyncio.iscoroutinefunction(chunk_handler):
                await chunk_handler(chunk)
            else:
                chunk_handler(chunk)

            logger.debug("Processed %d byte(s) of audio", len(chunk))

            # Update progress display
            seconds_streamed += len(chunk) / (config.PYAUDIO_RATE * config.PYAUDIO_CHANNELS * 2)
            if live and not quiet:
                if stop_event.ctrl_c_pressed:
                    msg = f"Ctrl+C pressed. Stopping {progress_message.lower()}..."
                    live.update(Text(msg, style="yellow"))
                else:
                    live.update(
                        Text(
                            f"{progress_message}... ({seconds_streamed:.1f}s)",
                            style=progress_style,
                        ),
                    )

    except OSError:
        logger.exception("Error reading audio")


def setup_input_stream(
    input_device_index: int | None,
) -> dict:
    """Get standard PyAudio input stream configuration.

    Args:
        p: PyAudio instance
        input_device_index: Input device index

    Returns:
        Dictionary of stream parameters

    """
    from agent_cli import config

    return {
        "format": config.PYAUDIO_FORMAT,
        "channels": config.PYAUDIO_CHANNELS,
        "rate": config.PYAUDIO_RATE,
        "input": True,
        "frames_per_buffer": config.PYAUDIO_CHUNK_SIZE,
        "input_device_index": input_device_index,
    }


def setup_output_stream(
    p: pyaudio.PyAudio,
    output_device_index: int | None,
    *,
    sample_rate: int | None = None,
    sample_width: int | None = None,
    channels: int | None = None,
) -> dict:
    """Get standard PyAudio output stream configuration.

    Args:
        p: PyAudio instance
        output_device_index: Output device index
        sample_rate: Custom sample rate (defaults to config)
        sample_width: Custom sample width in bytes (defaults to config)
        channels: Custom channel count (defaults to config)

    Returns:
        Dictionary of stream parameters

    """
    from agent_cli import config

    return {
        "format": p.get_format_from_width(sample_width or 2),
        "channels": channels or config.PYAUDIO_CHANNELS,
        "rate": sample_rate or config.PYAUDIO_RATE,
        "output": True,
        "frames_per_buffer": config.PYAUDIO_CHUNK_SIZE,
        "output_device_index": output_device_index,
    }


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


def get_device_by_index(p: pyaudio.PyAudio, input_device_index: int) -> dict:
    """Get device info by index from cached device list.

    Args:
        p: PyAudio instance
        input_device_index: Device index to look up

    Returns:
        Device info dictionary

    Raises:
        ValueError: If device index is not found

    """
    for device in get_all_devices(p):
        if device["index"] == input_device_index:
            return device
    msg = f"Device index {input_device_index} not found"
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
    input_device_name: str | None,
    input_device_index: int | None,
    key: str,
    what: str,
) -> tuple[int | None, str | None]:
    """Find an input device by a prioritized, comma-separated list of keywords."""
    if input_device_name is None and input_device_index is None:
        return None, None

    if input_device_index is not None:
        info = get_device_by_index(p, input_device_index)
        return input_device_index, info.get("name")
    assert input_device_name is not None
    search_terms = [term.strip().lower() for term in input_device_name.split(",") if term.strip()]

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

    msg = f"No {what} device found matching any of the keywords in {input_device_name!r}"
    raise ValueError(msg)


def input_device(
    p: pyaudio.PyAudio,
    input_device_name: str | None,
    input_device_index: int | None,
) -> tuple[int | None, str | None]:
    """Find an input device by a prioritized, comma-separated list of keywords."""
    return _in_or_out_device(p, input_device_name, input_device_index, "maxInputChannels", "input")


def output_device(
    p: pyaudio.PyAudio,
    input_device_name: str | None,
    input_device_index: int | None,
) -> tuple[int | None, str | None]:
    """Find an output device by a prioritized, comma-separated list of keywords."""
    return _in_or_out_device(
        p,
        input_device_name,
        input_device_index,
        "maxOutputChannels",
        "output",
    )
