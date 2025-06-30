"""Wyoming ASR Client for streaming microphone audio to a transcription server."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AbstractContextManager, nullcontext, suppress
from typing import TYPE_CHECKING

import pyperclip
from rich.live import Live
from rich.text import Text

import agent_cli.agents._cli_options as opts
from agent_cli import asr, process_manager
from agent_cli.agents._config import ASRConfig, GeneralConfig, LLMConfig
from agent_cli.audio import input_device, list_input_devices, pyaudio_context
from agent_cli.cli import app, setup_logging
from agent_cli.llm import process_and_update_clipboard
from agent_cli.utils import (
    print_device_index,
    print_input_panel,
    print_output_panel,
    print_status_message,
    signal_handling_context,
    stop_or_status,
)

if TYPE_CHECKING:
    import pyaudio
    from rich.console import Console

LOGGER = logging.getLogger()

SYSTEM_PROMPT = """
You are an AI transcription cleanup assistant. Your purpose is to improve and refine raw speech-to-text transcriptions by correcting errors, adding proper punctuation, and enhancing readability while preserving the original meaning and intent.

Your tasks include:
- Correcting obvious speech recognition errors and mishearing
- Adding appropriate punctuation (periods, commas, question marks, etc.)
- Fixing capitalization where needed
- Removing filler words, false starts, and repeated words when they clearly weren't intentional
- Improving sentence structure and flow while maintaining the speaker's voice and meaning
- Formatting the text for better readability

Important rules:
- Do not change the core meaning or content of the transcription
- Do not add information that wasn't spoken
- Do not remove content unless it's clearly an error or filler
- Return ONLY the cleaned-up text without any explanations or commentary
- Do not wrap your output in markdown or code blocks
"""

AGENT_INSTRUCTIONS = """
You will be given a block of raw transcribed text enclosed in <original-text> tags, and a cleanup instruction enclosed in <instruction> tags.

Your job is to process the transcribed text according to the instruction, which will typically involve:
- Correcting speech recognition errors
- Adding proper punctuation and capitalization
- Removing obvious filler words and false starts
- Improving readability while preserving meaning

Return ONLY the cleaned-up text with no additional formatting or commentary.
"""

INSTRUCTION = """
Please clean up this transcribed text by correcting any speech recognition errors, adding appropriate punctuation and capitalization, removing obvious filler words or false starts, and improving overall readability while preserving the original meaning and intent of the speaker.
"""


async def async_main(
    *,
    asr_config: ASRConfig,
    general_cfg: GeneralConfig,
    llm_config: LLMConfig,
    llm_enabled: bool,
    p: pyaudio.PyAudio,
) -> None:
    """Async entry point, consuming parsed args."""
    with (
        signal_handling_context(general_cfg.console, LOGGER) as stop_event,
        _maybe_live(general_cfg.console) as live,
    ):
        transcript = await asr.transcribe_audio(
            asr_server_ip=asr_config.server_ip,
            asr_server_port=asr_config.server_port,
            device_index=asr_config.device_index,
            logger=LOGGER,
            p=p,
            stop_event=stop_event,
            console=general_cfg.console,
            live=live,
            listening_message="Listening...",
        )

    if llm_enabled and llm_config.model and llm_config.ollama_host and transcript:
        print_input_panel(general_cfg.console, transcript, title="📝 Raw Transcript")
        await process_and_update_clipboard(
            system_prompt=SYSTEM_PROMPT,
            agent_instructions=AGENT_INSTRUCTIONS,
            model=llm_config.model,
            ollama_host=llm_config.ollama_host,
            logger=LOGGER,
            console=general_cfg.console,
            original_text=transcript,
            instruction=INSTRUCTION,
            clipboard=general_cfg.clipboard,
        )
        return

    # When not using LLM, show transcript in output panel for consistency
    if transcript:
        if general_cfg.quiet:
            # Quiet mode: print result to stdout for Keyboard Maestro to capture
            print(transcript)
        else:
            print_output_panel(
                general_cfg.console,
                transcript,
                title="📝 Transcript",
                subtitle="[dim]Copied to clipboard[/dim]" if general_cfg.clipboard else None,
            )

        if general_cfg.clipboard:
            pyperclip.copy(transcript)
            LOGGER.info("Copied transcript to clipboard.")
        else:
            LOGGER.info("Clipboard copy disabled.")
    else:
        LOGGER.info("Transcript empty.")
        if not general_cfg.quiet:
            print_status_message(
                general_cfg.console,
                "⚠️ No transcript captured.",
                style="yellow",
            )


def _maybe_live(console: Console | None) -> AbstractContextManager[Live | None]:
    if console:
        return Live(
            Text("Transcribing...", style="blue"),
            console=console,
            transient=True,
        )
    return nullcontext()


@app.command("transcribe")
def transcribe(
    *,
    device_index: int | None = opts.DEVICE_INDEX,
    device_name: str | None = opts.DEVICE_NAME,
    # ASR
    list_devices: bool = opts.LIST_DEVICES,
    asr_server_ip: str = opts.ASR_SERVER_IP,
    asr_server_port: int = opts.ASR_SERVER_PORT,
    # LLM
    model: str = opts.MODEL,
    ollama_host: str = opts.OLLAMA_HOST,
    llm: bool = opts.LLM,
    # Process control
    stop: bool = opts.STOP,
    status: bool = opts.STATUS,
    # General
    clipboard: bool = opts.CLIPBOARD,
    log_level: str = opts.LOG_LEVEL,
    log_file: str | None = opts.LOG_FILE,
    quiet: bool = opts.QUIET,
) -> None:
    """Wyoming ASR Client for streaming microphone audio to a transcription server.

    Usage:
    - Run in foreground: agent-cli transcribe --device-index 1
    - Run in background: agent-cli transcribe --device-index 1 &
    - Check status: agent-cli transcribe --status
    - Stop background process: agent-cli transcribe --stop
    """
    setup_logging(log_level, log_file, quiet=quiet)
    general_cfg = GeneralConfig(
        log_level=log_level,
        log_file=log_file,
        quiet=quiet,
        clipboard=clipboard,
    )
    process_name = "transcribe"
    if stop_or_status(process_name, "transcribe", general_cfg.console, stop, status):
        return

    with pyaudio_context() as p:
        if list_devices:
            list_input_devices(p, general_cfg.console)
            return
        device_index, device_name = input_device(p, device_name, device_index)
        print_device_index(general_cfg.console, device_index, device_name)

        # Use context manager for PID file management
        with process_manager.pid_file_context(process_name), suppress(KeyboardInterrupt):
            asr_config = ASRConfig(
                server_ip=asr_server_ip,
                server_port=asr_server_port,
                device_index=device_index,
                device_name=device_name,
                list_devices=list_devices,
            )
            llm_config = LLMConfig(model=model, ollama_host=ollama_host)

            asyncio.run(
                async_main(
                    asr_config=asr_config,
                    general_cfg=general_cfg,
                    llm_config=llm_config,
                    llm_enabled=llm,
                    p=p,
                ),
            )
