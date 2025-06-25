"""Tests for the transcribe agent."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assistant.agents import transcribe


@pytest.mark.asyncio
@patch("ai_assistant.agents.transcribe.cli")
@patch("ai_assistant.agents.transcribe.asr")
@patch("ai_assistant.agents.transcribe.pyperclip")
async def test_transcribe_main(
    mock_pyperclip: MagicMock,
    mock_asr: MagicMock,
    mock_cli: MagicMock,
) -> None:
    """Test the main function of the transcribe agent."""
    mock_args = MagicMock()
    mock_args.list_devices = False
    mock_args.clipboard = True
    mock_cli.get_base_parser.return_value.parse_args.return_value = mock_args

    mock_client = AsyncMock()
    mock_client.read_event.side_effect = [
        MagicMock(text="hello world"),
        asyncio.CancelledError,
    ]
    mock_asr.AsyncClient.from_uri.return_value.__aenter__.return_value = mock_client
    mock_asr.receive_text = AsyncMock(return_value="hello world")
    mock_asr.send_audio = AsyncMock()
    mock_asr.pyaudio_context.return_value.__enter__.return_value = MagicMock()

    await transcribe.async_main()

    mock_pyperclip.copy.assert_called_once_with("hello world")
