"""End-to-end tests for the voice assistant agent with minimal mocking."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import pytest
from rich.console import Console

from agent_cli.agents import voice_assistant
from tests.mocks.audio import MockPyAudio
from tests.mocks.llm import create_voice_assistant_responses, mock_build_agent
from tests.mocks.wyoming import MockWyomingAsyncClient


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.tts.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
@patch("agent_cli.llm.build_agent")
async def test_voice_assistant_basic_conversation(
    mock_build_agent_func,
    mock_pyaudio_class,
    mock_tts_client,
    mock_asr_client,
    mock_console: Console,
    mock_logger: logging.Logger,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test basic voice assistant conversation flow."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Setup mock LLM agent
    llm_responses = create_voice_assistant_responses()
    mock_agent = mock_build_agent(
        model="test-model",
        ollama_host="http://localhost:11434",
        responses=llm_responses,
    )
    mock_build_agent_func.return_value = mock_agent

    # Setup mock Wyoming clients
    mock_asr_client.from_uri = lambda uri, **kwargs: MockWyomingAsyncClient.from_uri(
        uri,
        asr_responses={"hello": "Hello there"},
        **kwargs,
    )

    mock_tts_client.from_uri = lambda uri, **kwargs: MockWyomingAsyncClient.from_uri(
        uri,
        tts_responses={"hello": synthetic_audio_data},
        **kwargs,
    )

    # Create a stop event that will be triggered after a short delay
    stop_event = asyncio.Event()

    async def trigger_stop():
        await asyncio.sleep(0.5)  # Short delay to allow conversation
        stop_event.set()

    # Start the stop trigger
    stop_task = asyncio.create_task(trigger_stop())

    with patch("agent_cli.agents.voice_assistant.pyperclip") as mock_pyperclip:
        # Mock clipboard content
        mock_pyperclip.paste.return_value = "This is some text to process"
        mock_pyperclip.copy = lambda x: None

        # Mock signal handling
        with patch(
            "agent_cli.agents.voice_assistant.signal_handling_context",
        ) as mock_signal_context:
            mock_signal_context.return_value.__enter__.return_value = stop_event
            mock_signal_context.return_value.__exit__.return_value = None

            try:
                # Run voice assistant
                await voice_assistant.async_main(
                    console=mock_console,
                    device_index=0,
                    device_name=None,
                    list_devices=False,
                    asr_server_ip="localhost",
                    asr_server_port=10300,
                    model="test-model",
                    ollama_host="http://localhost:11434",
                    clipboard=True,
                    enable_tts=True,
                    tts_server_ip="localhost",
                    tts_server_port=10200,
                    voice_name=None,
                    tts_language=None,
                    speaker=None,
                    output_device_index=None,
                    output_device_name=None,
                    list_output_devices_flag=False,
                    save_file=None,
                )

                # Verify LLM was called
                assert len(mock_agent.call_history) > 0

                # Verify audio streams were created
                assert len(mock_pyaudio.streams) > 0

                # Verify console output shows conversation
                console_output = mock_console.file.getvalue()
                assert (
                    "listening" in console_output.lower() or "assistant" in console_output.lower()
                )

            finally:
                # Clean up
                stop_task.cancel()
                try:
                    await stop_task
                except asyncio.CancelledError:
                    pass


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.tts.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
@patch("agent_cli.llm.build_agent")
async def test_voice_assistant_without_tts(
    mock_build_agent_func,
    mock_pyaudio_class,
    mock_tts_client,
    mock_asr_client,
    mock_console: Console,
    mock_logger: logging.Logger,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test voice assistant without TTS (text-only responses)."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Setup mock LLM agent
    llm_responses = create_voice_assistant_responses()
    mock_agent = mock_build_agent(
        model="test-model",
        ollama_host="http://localhost:11434",
        responses=llm_responses,
    )
    mock_build_agent_func.return_value = mock_agent

    # Setup mock ASR client
    mock_asr_client.from_uri = lambda uri, **kwargs: MockWyomingAsyncClient.from_uri(
        uri,
        asr_responses={"question": "What is the capital of France?"},
        **kwargs,
    )

    # Create a stop event
    stop_event = asyncio.Event()

    async def trigger_stop():
        await asyncio.sleep(0.3)
        stop_event.set()

    stop_task = asyncio.create_task(trigger_stop())

    try:
        # Run voice assistant without TTS
        await voice_assistant.async_main(
            asr_server_ip="localhost",
            asr_server_port=10300,
            device_index=0,
            enable_tts=False,  # No TTS
            tts_server_ip="localhost",
            tts_server_port=10200,
            voice_name=None,
            tts_language=None,
            speaker=None,
            output_device_index=None,
            output_device_name=None,
            list_output_devices_flag=False,
            save_file=None,
            model="test-model",
            ollama_host="http://localhost:11434",
            console=mock_console,
            p=mock_pyaudio,
            stop_event=stop_event,
        )

        # Verify LLM was called
        assert len(mock_agent.call_history) > 0

        # Verify only input streams were created (no TTS output streams)
        input_streams = [s for s in mock_pyaudio.streams if len(s.input_data) > 0]
        assert len(input_streams) > 0

        # Verify console shows text response
        console_output = mock_console.file.getvalue()
        assert "capital" in console_output.lower() or "france" in console_output.lower()

    finally:
        stop_task.cancel()
        try:
            await stop_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.tts.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
@patch("agent_cli.llm.build_agent")
async def test_voice_assistant_device_listing(
    mock_build_agent_func,
    mock_pyaudio_class,
    mock_tts_client,
    mock_asr_client,
    mock_console: Console,
    mock_logger: logging.Logger,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test voice assistant device listing functionality."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Run device listing (should not start conversation)
    await voice_assistant.async_main(
        asr_server_ip="localhost",
        asr_server_port=10300,
        device_index=0,
        enable_tts=True,
        tts_server_ip="localhost",
        tts_server_port=10200,
        voice_name=None,
        tts_language=None,
        speaker=None,
        output_device_index=None,
        output_device_name=None,
        list_output_devices_flag=True,  # List devices
        save_file=None,
        model="test-model",
        ollama_host="http://localhost:11434",
        console=mock_console,
        p=mock_pyaudio,
        stop_event=asyncio.Event(),
    )

    # Verify device list was displayed
    console_output = mock_console.file.getvalue()
    assert "Mock Output Device" in console_output
    assert "Mock Combined Device" in console_output

    # Should not have started conversation
    assert len(mock_pyaudio.streams) == 0


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.tts.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
@patch("agent_cli.llm.build_agent")
async def test_voice_assistant_with_tts_options(
    mock_build_agent_func,
    mock_pyaudio_class,
    mock_tts_client,
    mock_asr_client,
    mock_console: Console,
    mock_logger: logging.Logger,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test voice assistant with TTS voice options."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Setup mock LLM agent
    llm_responses = create_voice_assistant_responses()
    mock_agent = mock_build_agent(
        model="test-model",
        ollama_host="http://localhost:11434",
        responses=llm_responses,
    )
    mock_build_agent_func.return_value = mock_agent

    # Setup mock Wyoming clients
    mock_asr_client.from_uri = lambda uri, **kwargs: MockWyomingAsyncClient.from_uri(
        uri,
        asr_responses={"greeting": "Hello assistant"},
        **kwargs,
    )

    mock_tts_client.from_uri = lambda uri, **kwargs: MockWyomingAsyncClient.from_uri(
        uri,
        tts_responses={"hello": synthetic_audio_data},
        **kwargs,
    )

    # Create a stop event
    stop_event = asyncio.Event()

    async def trigger_stop():
        await asyncio.sleep(0.4)
        stop_event.set()

    stop_task = asyncio.create_task(trigger_stop())

    try:
        # Run voice assistant with TTS options
        await voice_assistant.async_main(
            asr_server_ip="localhost",
            asr_server_port=10300,
            device_index=0,
            tts_server_ip="localhost",
            tts_server_port=10200,
            voice_name="female_voice",
            tts_language="en-US",
            speaker="alice",
            output_device_index=1,
            output_device_name=None,
            list_output_devices_flag=False,
            save_file=None,
            model="test-model",
            ollama_host="http://localhost:11434",
            console=mock_console,
            p=mock_pyaudio,
            stop_event=stop_event,
        )

        # Verify conversation occurred
        assert len(mock_agent.call_history) > 0

        # Verify audio streams were created
        assert len(mock_pyaudio.streams) > 0

        # Verify output stream has data
        output_streams = [s for s in mock_pyaudio.streams if len(s.get_written_data()) > 0]
        assert len(output_streams) > 0

    finally:
        stop_task.cancel()
        try:
            await stop_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.tts.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
@patch("agent_cli.llm.build_agent")
async def test_voice_assistant_asr_error_handling(
    mock_build_agent_func,
    mock_pyaudio_class,
    mock_tts_client,
    mock_asr_client,
    mock_console: Console,
    mock_logger: logging.Logger,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test voice assistant handling of ASR connection errors."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Setup mock LLM agent
    llm_responses = create_voice_assistant_responses()
    mock_agent = mock_build_agent(
        model="test-model",
        ollama_host="http://localhost:11434",
        responses=llm_responses,
    )
    mock_build_agent_func.return_value = mock_agent

    # Mock ASR connection error
    async def mock_asr_error(uri, **kwargs):
        raise ConnectionRefusedError("ASR connection refused")

    mock_asr_client.from_uri = mock_asr_error

    # Run voice assistant - should handle error gracefully
    await voice_assistant.async_main(
        asr_server_ip="localhost",
        asr_server_port=10300,
        device_index=0,
        enable_tts=False,
        tts_server_ip="localhost",
        tts_server_port=10200,
        voice_name=None,
        tts_language=None,
        speaker=None,
        output_device_index=None,
        output_device_name=None,
        list_output_devices_flag=False,
        save_file=None,
        model="test-model",
        ollama_host="http://localhost:11434",
        console=mock_console,
        p=mock_pyaudio,
        stop_event=asyncio.Event(),
    )

    # Should not have started conversation due to ASR error
    assert len(mock_agent.call_history) == 0

    # Console should show error message
    console_output = mock_console.file.getvalue()
    assert "connection" in console_output.lower() or "error" in console_output.lower()


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.tts.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
@patch("agent_cli.llm.build_agent")
async def test_voice_assistant_quiet_mode(
    mock_build_agent_func,
    mock_pyaudio_class,
    mock_tts_client,
    mock_asr_client,
    mock_console: Console,
    mock_logger: logging.Logger,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test voice assistant in quiet mode."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Setup mock LLM agent
    llm_responses = create_voice_assistant_responses()
    mock_agent = mock_build_agent(
        model="test-model",
        ollama_host="http://localhost:11434",
        responses=llm_responses,
    )
    mock_build_agent_func.return_value = mock_agent

    # Setup mock Wyoming clients
    mock_asr_client.from_uri = lambda uri, **kwargs: MockWyomingAsyncClient.from_uri(
        uri,
        asr_responses={"quiet": "Testing quiet mode"},
        **kwargs,
    )

    mock_tts_client.from_uri = lambda uri, **kwargs: MockWyomingAsyncClient.from_uri(
        uri,
        tts_responses={"testing": synthetic_audio_data},
        **kwargs,
    )

    # Create a stop event
    stop_event = asyncio.Event()

    async def trigger_stop():
        await asyncio.sleep(0.3)
        stop_event.set()

    stop_task = asyncio.create_task(trigger_stop())

    try:
        # Run voice assistant in quiet mode
        await voice_assistant.async_main(
            asr_server_ip="localhost",
            asr_server_port=10300,
            device_index=0,
            tts_server_ip="localhost",
            tts_server_port=10200,
            voice_name=None,
            tts_language=None,
            speaker=None,
            output_device_index=None,
            output_device_name=None,
            list_output_devices_flag=False,
            save_file=None,
            model="test-model",
            ollama_host="http://localhost:11434",
            console=None,  # No console in quiet mode
            p=mock_pyaudio,
            stop_event=stop_event,
        )

        # Verify conversation still occurred
        assert len(mock_agent.call_history) > 0

        # Verify audio streams were created
        assert len(mock_pyaudio.streams) > 0

    finally:
        stop_task.cancel()
        try:
            await stop_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
@patch("agent_cli.asr.AsyncClient")
@patch("agent_cli.tts.AsyncClient")
@patch("agent_cli.audio.pyaudio.PyAudio")
@patch("agent_cli.llm.build_agent")
async def test_voice_assistant_multiple_interactions(
    mock_build_agent_func,
    mock_pyaudio_class,
    mock_tts_client,
    mock_asr_client,
    mock_console: Console,
    mock_logger: logging.Logger,
    synthetic_audio_data: bytes,
    mock_pyaudio_device_info: list[dict],
) -> None:
    """Test voice assistant with multiple conversation turns."""
    # Setup mock PyAudio
    mock_pyaudio = MockPyAudio(mock_pyaudio_device_info)
    mock_pyaudio_class.return_value = mock_pyaudio

    # Setup mock LLM agent with multiple responses
    llm_responses = create_voice_assistant_responses()
    mock_agent = mock_build_agent(
        model="test-model",
        ollama_host="http://localhost:11434",
        responses=llm_responses,
        simulate_delay=0.05,  # Fast responses for testing
    )
    mock_build_agent_func.return_value = mock_agent

    # Setup mock Wyoming clients
    conversation_turns = [
        "Hello assistant",
        "How are you?",
        "What can you do?",
        "Goodbye",
    ]

    turn_index = 0

    def create_asr_client(uri, **kwargs):
        nonlocal turn_index
        response = conversation_turns[turn_index % len(conversation_turns)]
        turn_index += 1
        return MockWyomingAsyncClient.from_uri(
            uri,
            asr_responses={"turn": response},
            simulate_delay=0.05,
            **kwargs,
        )

    mock_asr_client.from_uri = create_asr_client

    mock_tts_client.from_uri = lambda uri, **kwargs: MockWyomingAsyncClient.from_uri(
        uri,
        tts_responses={"response": synthetic_audio_data},
        simulate_delay=0.05,
        **kwargs,
    )

    # Create a stop event that triggers after allowing multiple turns
    stop_event = asyncio.Event()

    async def trigger_stop():
        await asyncio.sleep(1.0)  # Allow multiple conversation turns
        stop_event.set()

    stop_task = asyncio.create_task(trigger_stop())

    try:
        # Run voice assistant
        await voice_assistant.async_main(
            asr_server_ip="localhost",
            asr_server_port=10300,
            device_index=0,
            tts_server_ip="localhost",
            tts_server_port=10200,
            voice_name=None,
            tts_language=None,
            speaker=None,
            output_device_index=None,
            output_device_name=None,
            list_output_devices_flag=False,
            save_file=None,
            model="test-model",
            ollama_host="http://localhost:11434",
            console=mock_console,
            p=mock_pyaudio,
            stop_event=stop_event,
        )

        # Verify multiple LLM calls occurred
        assert len(mock_agent.call_history) >= 2

        # Verify multiple audio interactions
        assert len(mock_pyaudio.streams) >= 2

        # Verify conversation content
        console_output = mock_console.file.getvalue()
        assert "hello" in console_output.lower() or "how are you" in console_output.lower()

    finally:
        stop_task.cancel()
        try:
            await stop_task
        except asyncio.CancelledError:
            pass
