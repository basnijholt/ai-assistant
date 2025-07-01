r"""An interactive agent that you can talk to.

This agent will:
- Listen for your voice command.
- Transcribe the command.
- Send the transcription to an LLM.
- Speak the LLM's response.
- Remember the conversation history.
- Attach timestamps to the saved conversation.
- Format timestamps as "ago" when sending to the LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import typer

import agent_cli.agents._cli_options as opts
from agent_cli import asr, process_manager
from agent_cli.agents._config import (
    ASRConfig,
    FileConfig,
    GeneralConfig,
    LLMConfig,
    TTSConfig,
)
from agent_cli.agents._tts_common import handle_tts_playback
from agent_cli.audio import (
    input_device,
    list_input_devices,
    list_output_devices,
    output_device,
    pyaudio_context,
)
from agent_cli.cli import app, setup_logging
from agent_cli.llm import get_llm_response
from agent_cli.tools import ExecuteCodeTool, ReadFileTool
from agent_cli.utils import (
    InteractiveStopEvent,
    Timer,
    console,
    format_timedelta_to_ago,
    live_timer,
    maybe_live,
    print_device_index,
    print_input_panel,
    print_output_panel,
    print_status_message,
    signal_handling_context,
    stop_or_status,
)

if TYPE_CHECKING:
    import pyaudio
    from rich.live import Live


LOGGER = logging.getLogger(__name__)

# --- Conversation History ---


class ConversationEntry(TypedDict):
    """A single entry in the conversation."""

    role: str
    content: str
    timestamp: str


# --- LLM Prompts ---

SYSTEM_PROMPT = """\
You are a helpful and friendly conversational AI. Your role is to assist the user with their questions and tasks.

You have access to the following tools:
- read_file: Read the content of a file.
- execute_code: Execute a shell command.

- The user is interacting with you through voice, so keep your responses concise and natural.
- A summary of the previous conversation is provided for context. This context may or may not be relevant to the current query.
- Do not repeat information from the previous conversation unless it is necessary to answer the current question.
- Do not ask "How can I help you?" at the end of your response.
"""

AGENT_INSTRUCTIONS = """\
A summary of the previous conversation is provided in the <previous-conversation> tag.
The user's current message is in the <user-message> tag.

- If the user's message is a continuation of the previous conversation, use the context to inform your response.
- If the user's message is a new topic, ignore the previous conversation.

Your response should be helpful and directly address the user's message.
"""

USER_MESSAGE_WITH_CONTEXT_TEMPLATE = """
<previous-conversation>
{formatted_history}
</previous-conversation>
<user-message>
{instruction}
</user-message>
"""

# --- Helper Functions ---


def _setup_input_device(
    p: pyaudio.PyAudio,
    quiet: bool,
    device_name: str | None,
    device_index: int | None,
) -> tuple[int | None, str | None]:
    device_index, device_name = input_device(p, device_name, device_index)
    if not quiet:
        print_device_index(device_index, device_name)
    return device_index, device_name


def _setup_output_device(
    p: pyaudio.PyAudio,
    quiet: bool,
    device_name: str | None,
    device_index: int | None,
) -> tuple[int | None, str | None]:
    device_index, device_name = output_device(p, device_name, device_index)
    if not quiet and device_index is not None:
        msg = f"🔊 TTS output device [bold yellow]{device_index}[/bold yellow] ([italic]{device_name}[/italic])"
        print_status_message(msg)
    return device_index, device_name


def _load_conversation_history(history_file: Path) -> list[ConversationEntry]:
    if history_file.exists():
        with history_file.open("r") as f:
            return json.load(f)
    return []


def _save_conversation_history(history_file: Path, history: list[ConversationEntry]) -> None:
    with history_file.open("w") as f:
        json.dump(history, f, indent=2)


def _format_conversation_for_llm(history: list[ConversationEntry]) -> str:
    """Format the conversation history for the LLM."""
    if not history:
        return "No previous conversation."

    now = datetime.now(UTC)
    formatted_lines = []
    for entry in history:
        timestamp = datetime.fromisoformat(entry["timestamp"])
        ago = format_timedelta_to_ago(now - timestamp)
        formatted_lines.append(f"{entry['role']} ({ago}): {entry['content']}")
    return "\n".join(formatted_lines)


async def _handle_conversation_turn(
    *,
    p: pyaudio.PyAudio,
    stop_event: InteractiveStopEvent,
    conversation_history: list[ConversationEntry],
    general_cfg: GeneralConfig,
    asr_config: ASRConfig,
    llm_config: LLMConfig,
    tts_config: TTSConfig,
    file_config: FileConfig,
    live: Live,
) -> None:
    """Handles a single turn of the conversation."""
    # 1. Transcribe user's command
    instruction = await asr.transcribe_audio(
        asr_server_ip=asr_config.server_ip,
        asr_server_port=asr_config.server_port,
        device_index=asr_config.device_index,
        logger=LOGGER,
        p=p,
        stop_event=stop_event,
        quiet=general_cfg.quiet,
        live=live,
    )

    # Clear the stop event after ASR completes - it was only meant to stop recording
    stop_event.clear()

    if not instruction or not instruction.strip():
        if not general_cfg.quiet:
            print_status_message(
                "No instruction, listening again.",
                style="yellow",
            )
        return

    if not general_cfg.quiet:
        print_input_panel(instruction, title="👤 You")

    # 2. Add user message to history
    conversation_history.append(
        {
            "role": "user",
            "content": instruction,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )

    # 3. Format conversation for LLM
    formatted_history = _format_conversation_for_llm(conversation_history)
    user_message_with_context = USER_MESSAGE_WITH_CONTEXT_TEMPLATE.format(
        formatted_history=formatted_history,
        instruction=instruction,
    )

    # 4. Get LLM response with timing
    tools = [ReadFileTool, ExecuteCodeTool]
    timer = Timer()

    async with live_timer(
        live,
        f"🤖 Processing with {llm_config.model}",
        style="bold yellow",
        quiet=general_cfg.quiet,
        stop_event=stop_event,
    ):
        # Create a dummy Live for get_llm_response since we're using our own timer display
        response_text = await get_llm_response(
            system_prompt=SYSTEM_PROMPT,
            agent_instructions=AGENT_INSTRUCTIONS,
            user_input=user_message_with_context,
            model=llm_config.model,
            ollama_host=llm_config.ollama_host,
            logger=LOGGER,
            tools=tools,
            quiet=True,  # Suppress internal output since we're showing our own timer
            live=live,
        )

    elapsed_time = timer.stop()

    if not response_text:
        if not general_cfg.quiet:
            print_status_message("No response from LLM.", style="yellow")
        return

    if not general_cfg.quiet:
        print_output_panel(
            response_text,
            title="🤖 AI",
            subtitle=f"took {elapsed_time:.2f}s",
        )

    # 5. Add AI response to history
    conversation_history.append(
        {
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )

    # 6. Save history
    if file_config.history_dir:
        history_file = file_config.history_dir / "conversation.json"
        _save_conversation_history(history_file, conversation_history)

    # 7. Handle TTS playback
    if tts_config.enabled:
        await handle_tts_playback(
            response_text,
            tts_server_ip=tts_config.server_ip,
            tts_server_port=tts_config.server_port,
            voice_name=tts_config.voice_name,
            tts_language=tts_config.language,
            speaker=tts_config.speaker,
            output_device_index=tts_config.output_device_index,
            save_file=file_config.save_file,
            quiet=general_cfg.quiet,
            logger=LOGGER,
            play_audio=not file_config.save_file,
            stop_event=stop_event,
            speed=tts_config.speed,
            live=live,
        )

    # Reset stop_event for next iteration
    stop_event.clear()


# --- Main Application Logic ---


async def async_main(
    *,
    general_cfg: GeneralConfig,
    asr_config: ASRConfig,
    llm_config: LLMConfig,
    tts_config: TTSConfig,
    file_config: FileConfig,
) -> None:
    """Main async function, consumes parsed arguments."""
    try:
        with pyaudio_context() as p:
            # Handle device listing
            if asr_config.list_devices:
                list_input_devices(p, general_cfg.quiet)
                return

            if tts_config.list_output_devices:
                list_output_devices(p, general_cfg.quiet)
                return

            # Setup devices
            device_index, _ = _setup_input_device(
                p,
                general_cfg.quiet,
                asr_config.device_name,
                asr_config.device_index,
            )
            asr_config.device_index = device_index  # Update with the selected index

            if tts_config.enabled:
                tts_output_device_index, _ = _setup_output_device(
                    p,
                    general_cfg.quiet,
                    tts_config.output_device_name,
                    tts_config.output_device_index,
                )
                # Update with the selected index
                tts_config.output_device_index = tts_output_device_index

            # Load conversation history
            if file_config.history_dir:
                history_path = Path(file_config.history_dir).expanduser()
                history_path.mkdir(parents=True, exist_ok=True)
                history_file = history_path / "conversation.json"
                conversation_history = _load_conversation_history(history_file)

            with (
                maybe_live(not general_cfg.quiet) as live,
                signal_handling_context(LOGGER, general_cfg.quiet) as stop_event,
            ):
                while not stop_event.is_set():
                    await _handle_conversation_turn(
                        p=p,
                        stop_event=stop_event,
                        conversation_history=conversation_history,
                        general_cfg=general_cfg,
                        asr_config=asr_config,
                        llm_config=llm_config,
                        tts_config=tts_config,
                        file_config=file_config,
                        live=live,
                    )
    except Exception:
        if not general_cfg.quiet:
            console.print_exception()
        raise


@app.command("interactive")
def interactive(
    device_index: int | None = opts.DEVICE_INDEX,
    device_name: str | None = opts.DEVICE_NAME,
    *,
    # ASR
    list_devices: bool = opts.LIST_DEVICES,
    asr_server_ip: str = opts.ASR_SERVER_IP,
    asr_server_port: int = opts.ASR_SERVER_PORT,
    # LLM
    model: str = opts.MODEL,
    ollama_host: str = opts.OLLAMA_HOST,
    # Process control
    stop: bool = opts.STOP,
    status: bool = opts.STATUS,
    # General
    log_level: str = opts.LOG_LEVEL,
    log_file: str | None = opts.LOG_FILE,
    quiet: bool = opts.QUIET,
    # TTS parameters
    enable_tts: bool = opts.ENABLE_TTS,
    tts_server_ip: str = opts.TTS_SERVER_IP,
    tts_server_port: int = opts.TTS_SERVER_PORT,
    voice_name: str | None = opts.VOICE_NAME,
    tts_language: str | None = opts.TTS_LANGUAGE,
    speaker: str | None = opts.SPEAKER,
    tts_speed: float = opts.TTS_SPEED,
    output_device_index: int | None = opts.OUTPUT_DEVICE_INDEX,
    output_device_name: str | None = opts.OUTPUT_DEVICE_NAME,
    list_output_devices_flag: bool = opts.LIST_OUTPUT_DEVICES,
    # Output
    save_file: Path | None = opts.SAVE_FILE,
    # History
    history_dir: Path = typer.Option(  # noqa: B008
        "~/.config/agent-cli/history",
        "--history-dir",
        help="Directory to store conversation history.",
    ),
) -> None:
    """An interactive agent that you can talk to."""
    setup_logging(log_level, log_file, quiet=quiet)
    general_cfg = GeneralConfig(
        log_level=log_level,
        log_file=log_file,
        quiet=quiet,
        clipboard=False,  # Not used in interactive mode
    )
    process_name = "interactive"
    if stop_or_status(
        process_name,
        "interactive agent",
        stop,
        status,
        quiet=general_cfg.quiet,
    ):
        return

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
        tts_config = TTSConfig(
            enabled=enable_tts,
            server_ip=tts_server_ip,
            server_port=tts_server_port,
            voice_name=voice_name,
            language=tts_language,
            speaker=speaker,
            output_device_index=output_device_index,
            output_device_name=output_device_name,
            list_output_devices=list_output_devices_flag,
            speed=tts_speed,
        )
        file_config = FileConfig(save_file=save_file, history_dir=history_dir)

        asyncio.run(
            async_main(
                general_cfg=general_cfg,
                asr_config=asr_config,
                llm_config=llm_config,
                tts_config=tts_config,
                file_config=file_config,
            ),
        )
