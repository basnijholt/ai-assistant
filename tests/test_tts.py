"""Tests for the TTS module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_cli.tts import speak_text


@pytest.mark.asyncio
@patch("agent_cli.tts.synthesize_speech", new_callable=AsyncMock)
async def test_speak_text(mock_synthesize_speech: AsyncMock) -> None:
    """Test the speak_text function."""
    mock_synthesize_speech.return_value = b"audio data"
    audio_data = await speak_text(
        text="hello",
        tts_server_ip="localhost",
        tts_server_port=1234,
        voice_name="test-voice",
        language=None,
        speaker=None,
        output_device_index=None,
        play_audio_flag=False,
        logger=MagicMock(),
    )
    assert audio_data == b"audio data"
