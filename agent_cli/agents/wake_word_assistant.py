r"""Wake word-based voice assistant that records when wake word is detected.

This agent uses Wyoming wake word detection to implement a hands-free voice assistant that:
1. Continuously listens for a wake word
2. When the wake word is detected, starts recording user speech
3. When the wake word is detected again, stops recording and processes the speech
4. Sends the recorded speech to ASR for transcription
5. Optionally processes the transcript with an LLM and speaks the response

WORKFLOW:
1. Agent starts listening for the specified wake word
2. First wake word detection -> start recording user speech
3. Second wake word detection -> stop recording and process the speech
4. Transcribe the recorded speech using Wyoming ASR
5. Optionally process with LLM and respond with TTS

USAGE:
- Start the agent: wake-word-assistant --wake-word "ok_nabu" --input-device-index 1
- The agent runs continuously until stopped with Ctrl+C or --stop
- Uses background process management for daemon-like operation

REQUIREMENTS:
- Wyoming wake word server (e.g., wyoming-openwakeword)
- Wyoming ASR server (e.g., wyoming-whisper)
- Optional: Wyoming TTS server for responses
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import pyperclip
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient

import agent_cli.agents._cli_options as opts
from agent_cli import asr, config, process_manager, wake_word
from agent_cli.agents._config import (
    ASRConfig,
    FileConfig,
    GeneralConfig,
    LLMConfig,
    TTSConfig,
    WakeWordConfig,
)
from agent_cli.agents._tts_common import handle_tts_playback
from agent_cli.audio import (
    input_device,
    list_input_devices,
    list_output_devices,
    open_pyaudio_stream,
    output_device,
    pyaudio_context,
)
from agent_cli.cli import app, setup_logging
from agent_cli.llm import process_and_update_clipboard
from agent_cli.tts import _create_wav_data  # Reuse existing WAV creation
from agent_cli.utils import (
    InteractiveStopEvent,
    console,
    maybe_live,
    print_device_index,
    print_input_panel,
    print_with_style,
    signal_handling_context,
    stop_or_status_or_toggle,
)

if TYPE_CHECKING:
    import pyaudio
    from rich.live import Live

LOGGER = logging.getLogger()

# LLM Prompts for wake word assistant
SYSTEM_PROMPT = """\
You are a helpful voice assistant. Respond to user questions and commands in a conversational, friendly manner.

Keep your responses concise but informative. If the user asks you to perform an action that requires external tools or systems, explain what you would do if you had access to those capabilities.

Always be helpful, accurate, and engaging in your responses.
"""

AGENT_INSTRUCTIONS = """\
The user has spoken a voice command or question. Provide a helpful, conversational response.

If it's a question, answer it clearly and concisely.
If it's a command, explain what you would do or provide guidance on how to accomplish it.
If it's unclear, ask for clarification in a friendly way.

Respond as if you're having a natural conversation.
"""


async def record_audio_to_buffer(
    p: pyaudio.PyAudio,
    input_device_index: int | None,
    stop_event: InteractiveStopEvent,
    logger: logging.Logger,
    *,
    quiet: bool = False,
    live: Live | None = None,
) -> bytes:
    """Record audio to a buffer using ASR module pattern.

    Args:
        p: PyAudio instance
        input_device_index: Audio input device index
        stop_event: Event to stop recording
        logger: Logger instance
        quiet: If True, suppress console output
        live: Rich Live display for progress

    Returns:
        Raw audio data as bytes

    """
    if not quiet:
        print_with_style("🎤 Recording... Say the wake word again to stop", style="green")

    # Use ASR module's audio recording functionality
    with open_pyaudio_stream(
        p,
        format=config.PYAUDIO_FORMAT,
        channels=config.PYAUDIO_CHANNELS,
        rate=config.PYAUDIO_RATE,
        input=True,
        frames_per_buffer=config.PYAUDIO_CHUNK_SIZE,
        input_device_index=input_device_index,
    ) as stream:
        return await asr.record_audio_to_buffer(
            stream=stream,
            stop_event=stop_event,
            logger=logger,
            live=live,
            quiet=quiet,
            progress_message="Recording",
            progress_style="green",
        )


async def save_audio_as_wav(audio_data: bytes, filename: str) -> None:
    """Save raw audio data as WAV file using existing TTS functionality.

    Args:
        audio_data: Raw audio bytes
        filename: Output filename

    """
    # Reuse the WAV creation logic from TTS module
    wav_data = _create_wav_data(
        audio_data,
        sample_rate=config.PYAUDIO_RATE,
        sample_width=2,  # 16-bit audio
        channels=config.PYAUDIO_CHANNELS,
    )

    # Write the WAV data to file asynchronously
    await asyncio.to_thread(Path(filename).write_bytes, wav_data)


async def _process_recorded_audio(
    audio_data: bytes,
    asr_server_ip: str,
    asr_server_port: int,
    logger: logging.Logger,
) -> str:
    """Process pre-recorded audio data with Wyoming ASR server.

    Args:
        audio_data: Raw audio bytes
        asr_server_ip: Wyoming ASR server IP
        asr_server_port: Wyoming ASR server port
        logger: Logger instance

    Returns:
        Transcribed text

    Raises:
        Exception: If ASR processing fails

    """
    from agent_cli.wyoming_utils import wyoming_client_context
    
    async with wyoming_client_context(asr_server_ip, asr_server_port, "ASR", logger) as client:
        # Start transcription
        await client.write_event(Transcribe().event())
        await client.write_event(AudioStart(**config.WYOMING_AUDIO_CONFIG).event())

        # Send audio data in chunks
        chunk_size = config.PYAUDIO_CHUNK_SIZE * 2  # 2 bytes per sample for 16-bit
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            await client.write_event(
                AudioChunk(audio=chunk, **config.WYOMING_AUDIO_CONFIG).event(),
            )
            logger.debug("Sent %d byte(s) of audio", len(chunk))

        # Signal end of audio
        await client.write_event(AudioStop().event())
        logger.debug("Sent AudioStop")

        # Receive transcript
        while True:
            event = await client.read_event()
            if event is None:
                logger.warning("Connection to ASR server lost.")
                break

            if Transcript.is_type(event.type):
                transcript = Transcript.from_event(event)
                logger.info("Final transcript: %s", transcript.text)
                return transcript.text

        return ""


async def async_main(
    *,
    general_cfg: GeneralConfig,
    wake_word_config: WakeWordConfig,
    asr_config: ASRConfig,
    llm_config: LLMConfig,
    tts_config: TTSConfig,
    file_config: FileConfig,
) -> None:
    """Main async function for the wake word assistant."""
    with pyaudio_context() as p:
        # Handle device listing
        if wake_word_config.list_input_devices:
            list_input_devices(p, not general_cfg.quiet)
            return

        if tts_config.list_output_devices:
            list_output_devices(p, not general_cfg.quiet)
            return

        # Setup input device
        input_device_index, input_device_name = input_device(
            p,
            wake_word_config.input_device_name,
            wake_word_config.input_device_index,
        )
        if not general_cfg.quiet:
            print_device_index(input_device_index, input_device_name)

        # Setup output device for TTS if enabled
        tts_output_device_index = tts_config.output_device_index
        if tts_config.enabled and (tts_config.output_device_name or tts_config.output_device_index):
            tts_output_device_index, tts_output_device_name = output_device(
                p,
                tts_config.output_device_name,
                tts_config.output_device_index,
            )
            if tts_output_device_index is not None and not general_cfg.quiet:
                msg = f"🔊 TTS output device [bold yellow]{tts_output_device_index}[/bold yellow] ([italic]{tts_output_device_name}[/italic])"
                print_with_style(msg)

        if not general_cfg.quiet:
            wake_word_msg = f"👂 Listening for wake word: [bold yellow]{wake_word_config.wake_word_name}[/bold yellow]"
            print_with_style(wake_word_msg)
            print_with_style("Say the wake word to start recording, then say it again to stop and process.", style="dim")

        with (
            maybe_live(not general_cfg.quiet) as live,
            signal_handling_context(LOGGER, general_cfg.quiet) as main_stop_event,
        ):
            while not main_stop_event.is_set():
                # Listen for first wake word detection (start recording)
                if not general_cfg.quiet:
                    console.print("🔍 Waiting for wake word to start recording...")

                detected_word = await wake_word.detect_wake_word(
                    wake_server_ip=wake_word_config.server_ip,
                    wake_server_port=wake_word_config.server_port,
                    wake_word_name=wake_word_config.wake_word_name,
                    input_device_index=input_device_index,
                    logger=LOGGER,
                    p=p,
                    stop_event=main_stop_event,
                    live=live,
                    quiet=general_cfg.quiet,
                )

                if not detected_word or main_stop_event.is_set():
                    break

                if not general_cfg.quiet:
                    print_with_style(f"✅ Wake word '{detected_word}' detected! Starting recording...", style="green")

                # Create a new stop event for recording
                recording_stop_event = InteractiveStopEvent()

                # Start recording in the background
                record_task = asyncio.create_task(
                    record_audio_to_buffer(
                        p,
                        input_device_index,
                        recording_stop_event,
                        LOGGER,
                        quiet=general_cfg.quiet,
                        live=live,
                    ),
                )

                # Listen for second wake word detection (stop recording)
                stop_detected_word = await wake_word.detect_wake_word(
                    wake_server_ip=wake_word_config.server_ip,
                    wake_server_port=wake_word_config.server_port,
                    wake_word_name=wake_word_config.wake_word_name,
                    input_device_index=input_device_index,
                    logger=LOGGER,
                    p=p,
                    stop_event=main_stop_event,
                    live=live,
                    quiet=general_cfg.quiet,
                )

                # Stop recording
                recording_stop_event.set()
                audio_data = await record_task

                if not stop_detected_word or main_stop_event.is_set():
                    break

                if not general_cfg.quiet:
                    print_with_style(f"🛑 Wake word '{stop_detected_word}' detected! Stopping recording...", style="yellow")

                if not audio_data:
                    if not general_cfg.quiet:
                        print_with_style("No audio recorded", style="yellow")
                    continue

                # Save recorded audio for debugging (optional)
                if file_config.save_file:
                    await save_audio_as_wav(audio_data, str(file_config.save_file))
                    if not general_cfg.quiet:
                        print_with_style(f"💾 Audio saved to {file_config.save_file}", style="blue")

                # Process recorded audio with ASR
                if not general_cfg.quiet:
                    print_with_style("🔄 Processing recorded audio...", style="blue")

                try:
                    # Send audio data to Wyoming ASR server for transcription
                    transcript = await _process_recorded_audio(
                        audio_data,
                        asr_server_ip=asr_config.server_ip,
                        asr_server_port=asr_config.server_port,
                        logger=LOGGER,
                    )

                    if not transcript or not transcript.strip():
                        if not general_cfg.quiet:
                            print_with_style("No speech detected in recording", style="yellow")
                        continue

                except Exception as e:
                    LOGGER.exception("Failed to process audio with ASR")
                    if not general_cfg.quiet:
                        print_with_style(f"ASR processing failed: {e}", style="red")
                    continue

                if not general_cfg.quiet:
                    print_input_panel(
                        transcript,
                        title="🎯 Transcribed Speech",
                        style="bold yellow",
                    )

                # Process with LLM if clipboard mode is enabled
                if general_cfg.clipboard:
                    await process_and_update_clipboard(
                        system_prompt=SYSTEM_PROMPT,
                        agent_instructions=AGENT_INSTRUCTIONS,
                        model=llm_config.model,
                        ollama_host=llm_config.ollama_host,
                        logger=LOGGER,
                        original_text="",  # No original text for voice assistant
                        instruction=transcript,
                        clipboard=general_cfg.clipboard,
                        quiet=general_cfg.quiet,
                        live=live,
                    )

                    # Handle TTS response if enabled
                    if tts_config.enabled:
                        response_text = pyperclip.paste()
                        if response_text and response_text.strip():
                            await handle_tts_playback(
                                response_text,
                                tts_server_ip=tts_config.server_ip,
                                tts_server_port=tts_config.server_port,
                                voice_name=tts_config.voice_name,
                                tts_language=tts_config.language,
                                speaker=tts_config.speaker,
                                output_device_index=tts_output_device_index,
                                save_file=file_config.save_file,
                                quiet=general_cfg.quiet,
                                logger=LOGGER,
                                play_audio=not file_config.save_file,
                                status_message="🔊 Speaking response...",
                                description="TTS audio",
                                speed=tts_config.speed,
                                live=live,
                            )

                if not general_cfg.quiet:
                    print_with_style("✨ Ready for next wake word...", style="green")


@app.command("wake-word-assistant")
def wake_word_assistant(
    *,
    # Wake word parameters
    wake_server_ip: str = opts.WAKE_WORD_SERVER_IP,
    wake_server_port: int = opts.WAKE_WORD_SERVER_PORT,
    wake_word: str = opts.WAKE_WORD_NAME,
    # ASR parameters
    input_device_index: int | None = opts.DEVICE_INDEX,
    input_device_name: str | None = opts.DEVICE_NAME,
    list_input_devices: bool = opts.LIST_DEVICES,
    asr_server_ip: str = opts.ASR_SERVER_IP,
    asr_server_port: int = opts.ASR_SERVER_PORT,
    # LLM parameters
    model: str = opts.MODEL,
    ollama_host: str = opts.OLLAMA_HOST,
    # Process control
    stop: bool = opts.STOP,
    status: bool = opts.STATUS,
    toggle: bool = opts.TOGGLE,
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
    # General
    clipboard: bool = opts.CLIPBOARD,
    log_level: str = opts.LOG_LEVEL,
    log_file: str | None = opts.LOG_FILE,
    quiet: bool = opts.QUIET,
    config_file: str | None = opts.CONFIG_FILE,  # noqa: ARG001
) -> None:
    """Wake word-based voice assistant using Wyoming wake word detection.

    This agent continuously listens for a wake word. When detected, it starts recording
    audio until the wake word is said again, then processes the recorded speech.

    Usage:
    - Start: agent-cli wake-word-assistant --wake-word "ok_nabu" --input-device-index 1
    - With TTS: agent-cli wake-word-assistant --wake-word "ok_nabu" --tts --voice "en_US-lessac-medium"
    - Background: agent-cli wake-word-assistant --wake-word "ok_nabu" &
    - Stop: agent-cli wake-word-assistant --stop
    - Status: agent-cli wake-word-assistant --status

    Requirements:
    - Wyoming wake word server (e.g., wyoming-openwakeword on port 10400)
    - Wyoming ASR server (e.g., wyoming-whisper)
    - Optional: Wyoming TTS server for responses
    """
    setup_logging(log_level, log_file, quiet=quiet)
    general_cfg = GeneralConfig(
        log_level=log_level,
        log_file=log_file,
        quiet=quiet,
        clipboard=clipboard,
    )
    process_name = "wake-word-assistant"
    if stop_or_status_or_toggle(
        process_name,
        "wake word assistant",
        stop,
        status,
        toggle,
        quiet=general_cfg.quiet,
    ):
        return

    # Use context manager for PID file management
    with process_manager.pid_file_context(process_name), suppress(KeyboardInterrupt):
        wake_word_config = WakeWordConfig(
            server_ip=wake_server_ip,
            server_port=wake_server_port,
            wake_word_name=wake_word,
            input_device_index=input_device_index,
            input_device_name=input_device_name,
            list_input_devices=list_input_devices,
        )
        asr_config = ASRConfig(
            server_ip=asr_server_ip,
            server_port=asr_server_port,
            input_device_index=input_device_index,
            input_device_name=input_device_name,
            list_input_devices=list_input_devices,
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
        file_config = FileConfig(save_file=save_file)

        asyncio.run(
            async_main(
                general_cfg=general_cfg,
                wake_word_config=wake_word_config,
                asr_config=asr_config,
                llm_config=llm_config,
                tts_config=tts_config,
                file_config=file_config,
            ),
        )
