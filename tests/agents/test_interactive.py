"""Tests for the interactive agent."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_cli.agents.interactive import (
    ConversationEntry,
    _format_conversation_for_llm,
    _load_conversation_history,
    _save_conversation_history,
    async_main,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def history_file(tmp_path: Path) -> Path:
    """Create a temporary history file."""
    return tmp_path / "conversation.json"


def test_load_and_save_conversation_history(history_file: Path) -> None:
    """Test saving and loading conversation history."""
    # 1. Test loading from a non-existent file
    history = _load_conversation_history(history_file)
    assert history == []

    # 2. Test saving and then loading
    now = datetime.now(UTC).isoformat()
    history_to_save: list[ConversationEntry] = [
        {"role": "user", "content": "Hello", "timestamp": now},
        {"role": "assistant", "content": "Hi there!", "timestamp": now},
    ]
    _save_conversation_history(history_file, history_to_save)

    loaded_history = _load_conversation_history(history_file)
    assert loaded_history == history_to_save


def test_format_conversation_for_llm() -> None:
    """Test formatting conversation history for the LLM."""
    # 1. Test with no history
    assert _format_conversation_for_llm([]) == "No previous conversation."

    # 2. Test with history
    now = datetime.now(UTC)
    history: list[ConversationEntry] = [
        {
            "role": "user",
            "content": "What's the weather?",
            "timestamp": (now - timedelta(minutes=5)).isoformat(),
        },
        {
            "role": "assistant",
            "content": "It's sunny.",
            "timestamp": (now - timedelta(minutes=4)).isoformat(),
        },
    ]
    formatted = _format_conversation_for_llm(history)
    assert "user (5 minutes ago): What's the weather?" in formatted
    assert "assistant (4 minutes ago): It's sunny." in formatted


@pytest.mark.asyncio
async def test_async_main_list_devices(tmp_path: Path) -> None:
    """Test the async_main function with list_devices=True."""
    with (
        patch("agent_cli.agents.interactive.pyaudio_context"),
        patch(
            "agent_cli.agents.interactive.list_input_devices",
        ) as mock_list_input_devices,
    ):
        await async_main(
            console=MagicMock(),
            device_index=None,
            device_name=None,
            list_devices=True,
            asr_server_ip="localhost",
            asr_server_port=1234,
            model="test-model",
            ollama_host="localhost",
            enable_tts=False,
            tts_server_ip="localhost",
            tts_server_port=5678,
            voice_name=None,
            tts_language=None,
            speaker=None,
            output_device_index=None,
            output_device_name=None,
            list_output_devices_flag=False,
            save_file=None,
            history_dir=str(tmp_path),
        )
        mock_list_input_devices.assert_called_once()


@pytest.mark.asyncio
async def test_async_main_list_output_devices(tmp_path: Path) -> None:
    """Test the async_main function with list_output_devices_flag=True."""
    with (
        patch("agent_cli.agents.interactive.pyaudio_context"),
        patch(
            "agent_cli.agents.interactive.list_output_devices",
        ) as mock_list_output_devices,
    ):
        await async_main(
            console=MagicMock(),
            device_index=None,
            device_name=None,
            list_devices=False,
            asr_server_ip="localhost",
            asr_server_port=1234,
            model="test-model",
            ollama_host="localhost",
            enable_tts=False,
            tts_server_ip="localhost",
            tts_server_port=5678,
            voice_name=None,
            tts_language=None,
            speaker=None,
            output_device_index=None,
            output_device_name=None,
            list_output_devices_flag=True,
            save_file=None,
            history_dir=str(tmp_path),
        )
        mock_list_output_devices.assert_called_once()


@pytest.mark.asyncio
async def test_async_main_full_loop(tmp_path: Path) -> None:
    """Test a full loop of the interactive agent's async_main function."""
    history_dir = tmp_path / "history"
    history_dir.mkdir()

    with (
        patch("agent_cli.agents.interactive.pyaudio_context"),
        patch("agent_cli.agents.interactive._setup_input_device", return_value=(1, "mock_input")),
        patch("agent_cli.agents.interactive._setup_output_device", return_value=(1, "mock_output")),
        patch(
            "agent_cli.agents.interactive.asr.transcribe_audio",
            new_callable=AsyncMock,
        ) as mock_transcribe,
        patch(
            "agent_cli.agents.interactive.get_llm_response",
            new_callable=AsyncMock,
        ) as mock_llm_response,
        patch(
            "agent_cli.agents.interactive.handle_tts_playback",
            new_callable=AsyncMock,
        ) as mock_tts,
        patch("agent_cli.agents.interactive.signal_handling_context") as mock_signal,
    ):
        # Simulate a single loop by setting the stop event after the first transcription
        stop_event = asyncio.Event()

        async def transcribe_and_stop(*_args: object, **_kwargs: object) -> str:
            """Set the stop event after one call and return a mock instruction."""
            stop_event.set()
            return "Mocked instruction"

        mock_transcribe.side_effect = transcribe_and_stop
        mock_llm_response.return_value = "Mocked response"
        mock_signal.return_value.__enter__.return_value = stop_event

        await async_main(
            console=MagicMock(),
            device_index=1,
            device_name=None,
            list_devices=False,
            asr_server_ip="localhost",
            asr_server_port=1234,
            model="test-model",
            ollama_host="localhost",
            enable_tts=True,
            tts_server_ip="localhost",
            tts_server_port=5678,
            voice_name="test-voice",
            tts_language="en",
            speaker=None,
            output_device_index=1,
            output_device_name=None,
            list_output_devices_flag=False,
            save_file=None,
            history_dir=str(history_dir),
        )

        # Verify that the core functions were called
        mock_transcribe.assert_called()
        mock_llm_response.assert_called()
        mock_tts.assert_called_with(
            "Mocked response",
            tts_server_ip="localhost",
            tts_server_port=5678,
            voice_name="test-voice",
            tts_language="en",
            speaker=None,
            output_device_index=1,
            save_file=None,
            console=mock_tts.call_args[1]["console"],
            logger=mock_tts.call_args[1]["logger"],
            play_audio=True,
        )

        # Verify that history was saved
        history_file = history_dir / "conversation.json"
        assert history_file.exists()
        with history_file.open("r") as f:
            history = json.load(f)

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Mocked instruction"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Mocked response"
