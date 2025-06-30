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
from rich.console import Console

import agent_cli.agents._cli_options as opts
from agent_cli import asr, process_manager
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
    format_timedelta_to_ago,
    print_device_index,
    print_input_panel,
    print_status_message,
    signal_handling_context,
)

if TYPE_CHECKING:
    import pyaudio


LOGGER = logging.getLogger()

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
    console: Console | None,
    device_name: str | None,
    device_index: int | None,
) -> tuple[int | None, str | None]:
    device_index, device_name = input_device(p, device_name, device_index)
    print_device_index(console, device_index, device_name)
    return device_index, device_name


def _setup_output_device(
    p: pyaudio.PyAudio,
    console: Console | None,
    device_name: str | None,
    device_index: int | None,
) -> tuple[int | None, str | None]:
    device_index, device_name = output_device(p, device_name, device_index)
    if device_index is not None and console:
        msg = f"🔊 TTS output device [bold yellow]{device_index}[/bold yellow] ([italic]{device_name}[/italic])"
        print_status_message(console, msg)
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
    if not history:
        return "No previous conversation."

    now = datetime.now(UTC)
    formatted_lines = []
    for entry in history:
        timestamp = datetime.fromisoformat(entry["timestamp"])
        ago = format_timedelta_to_ago(now - timestamp)
        formatted_lines.append(f"{entry['role']} ({ago}): {entry['content']}")
    return "\n".join(formatted_lines)


# --- Main Application Logic ---


async def async_main(
    *,
    # General
    console: Console | None,
    # ASR input device
    device_index: int | None,
    device_name: str | None,
    list_devices: bool,
    # ASR parameters
    asr_server_ip: str,
    asr_server_port: int,
    # LLM parameters
    model: str,
    ollama_host: str,
    # TTS parameters
    enable_tts: bool,
    tts_server_ip: str,
    tts_server_port: int,
    voice_name: str | None,
    tts_language: str | None,
    speaker: str | None,
    # Output device
    output_device_index: int | None,
    output_device_name: str | None,
    list_output_devices_flag: bool,
    # Output file
    save_file: str | None,
    # History
    history_dir: str,
) -> None:
    """Main async function, consumes parsed arguments."""
    with pyaudio_context() as p:
        # Handle device listing
        if list_devices:
            list_input_devices(p, console)
            return

        if list_output_devices_flag:
            list_output_devices(p, console)
            return

        # Setup devices
        device_index, _ = _setup_input_device(p, console, device_name, device_index)
        tts_output_device_index = output_device_index
        if enable_tts:
            tts_output_device_index, _ = _setup_output_device(
                p,
                console,
                output_device_name,
                output_device_index,
            )

        # Load conversation history
        history_path = Path(history_dir).expanduser()
        history_path.mkdir(parents=True, exist_ok=True)
        history_file = history_path / "conversation.json"
        conversation_history = _load_conversation_history(history_file)

        with signal_handling_context(console, LOGGER) as stop_event:
            while not stop_event.is_set():
                # 1. Transcribe user's command
                print_status_message(console, "Listening for your command...", style="bold cyan")
                instruction = await asr.transcribe_audio(
                    asr_server_ip=asr_server_ip,
                    asr_server_port=asr_server_port,
                    device_index=device_index,
                    logger=LOGGER,
                    p=p,
                    stop_event=stop_event,
                    console=console,
                )

                if not instruction or not instruction.strip():
                    print_status_message(
                        console,
                        "No instruction, listening again.",
                        style="yellow",
                    )
                    continue

                print_input_panel(console, instruction, title="You")

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

                # 4. Get LLM response
                tools = [ReadFileTool, ExecuteCodeTool]
                response_text = await get_llm_response(
                    system_prompt=SYSTEM_PROMPT,
                    agent_instructions=AGENT_INSTRUCTIONS,
                    user_input=user_message_with_context,
                    model=model,
                    ollama_host=ollama_host,
                    logger=LOGGER,
                    console=console,
                    tools=tools,
                )

                if not response_text:
                    print_status_message(console, "No response from LLM.", style="yellow")
                    continue

                print_input_panel(console, response_text, title="AI")

                # 5. Add AI response to history
                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

                # 6. Save history
                _save_conversation_history(history_file, conversation_history)

                # 7. Handle TTS playback
                if enable_tts:
                    await handle_tts_playback(
                        response_text,
                        tts_server_ip=tts_server_ip,
                        tts_server_port=tts_server_port,
                        voice_name=voice_name,
                        tts_language=tts_language,
                        speaker=speaker,
                        output_device_index=tts_output_device_index,
                        save_file=save_file,
                        console=console,
                        logger=LOGGER,
                        play_audio=not save_file,
                    )

                # 8. Reset stop_event for next iteration
                stop_event.clear()


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
    output_device_index: int | None = opts.OUTPUT_DEVICE_INDEX,
    output_device_name: str | None = opts.OUTPUT_DEVICE_NAME,
    list_output_devices_flag: bool = opts.LIST_OUTPUT_DEVICES,
    # Output
    save_file: str | None = typer.Option(
        None,
        "--save-file",
        help="Save TTS response audio to WAV file.",
    ),
    # History
    history_dir: str = typer.Option(
        "~/.config/agent-cli/history",
        "--history-dir",
        help="Directory to store conversation history.",
    ),
) -> None:
    """An interactive agent that you can talk to."""
    setup_logging(log_level, log_file, quiet=quiet)
    console = Console() if not quiet else None
    process_name = "interactive"

    if stop:
        if process_manager.kill_process(process_name):
            print_status_message(console, "✅ Interactive agent stopped.")
        else:
            print_status_message(console, "⚠️  No interactive agent is running.", style="yellow")
        return

    if status:
        if process_manager.is_process_running(process_name):
            pid = process_manager.read_pid_file(process_name)
            print_status_message(console, f"✅ Interactive agent is running (PID: {pid}).")
        else:
            print_status_message(console, "⚠️  Interactive agent is not running.", style="yellow")
        return

    # Use context manager for PID file management
    with process_manager.pid_file_context(process_name), suppress(KeyboardInterrupt):
        asyncio.run(
            async_main(
                # General
                console=console,
                # ASR input device
                device_index=device_index,
                device_name=device_name,
                list_devices=list_devices,
                # ASR parameters
                asr_server_ip=asr_server_ip,
                asr_server_port=asr_server_port,
                # LLM parameters
                model=model,
                ollama_host=ollama_host,
                # TTS parameters
                enable_tts=enable_tts,
                tts_server_ip=tts_server_ip,
                tts_server_port=tts_server_port,
                voice_name=voice_name,
                tts_language=tts_language,
                speaker=speaker,
                # Output device
                output_device_index=output_device_index,
                output_device_name=output_device_name,
                list_output_devices_flag=list_output_devices_flag,
                # Output file
                save_file=save_file,
                # History
                history_dir=history_dir,
            ),
        )
