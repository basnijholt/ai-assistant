"""Tests for the TTS common module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from agent_cli.agents._tts_common import handle_tts_playback


@pytest.mark.asyncio
@patch("agent_cli.agents._tts_common.tts.speak_text", new_callable=AsyncMock)
async def test_handle_tts_playback(mock_speak_text: AsyncMock) -> None:
    """Test the handle_tts_playback function."""
    mock_speak_text.return_value = b"audio data"
    await handle_tts_playback(
        text="hello",
        tts_server_ip="localhost",
        tts_server_port=1234,
        voice_name="test-voice",
        tts_language="en",
        speaker=None,
        output_device_index=1,
        save_file=None,
        quiet=False,
        logger=MagicMock(),
        play_audio=True,
        speed=1.0,
    )

    mock_speak_text.assert_called_once_with(
        text="hello",
        tts_server_ip="localhost",
        tts_server_port=1234,
        logger=ANY,
        voice_name="test-voice",
        language="en",
        speaker=None,
        output_device_index=1,
        quiet=False,
        play_audio_flag=True,
        stop_event=None,
        speed=1.0,
    )

    # Test with save_file
    with patch("pathlib.Path.write_bytes") as mock_write_bytes:
        await handle_tts_playback(
            text="hello",
            tts_server_ip="localhost",
            tts_server_port=1234,
            voice_name="test-voice",
            tts_language="en",
            speaker=None,
            output_device_index=1,
            save_file=Path("test.wav"),
            quiet=False,
            logger=MagicMock(),
            play_audio=False,
            speed=1.0,
        )
        mock_write_bytes.assert_called_once_with(b"audio data")
    assert mock_speak_text.call_count == 2


@pytest.mark.asyncio
@patch("agent_cli.agents._tts_common.tts.speak_text", new_callable=AsyncMock)
async def test_handle_tts_playback_no_audio(mock_speak_text: AsyncMock) -> None:
    """Test the handle_tts_playback function when no audio is returned."""
    mock_speak_text.return_value = None
    await handle_tts_playback(
        text="hello",
        tts_server_ip="localhost",
        tts_server_port=1234,
        voice_name="test-voice",
        tts_language="en",
        speaker=None,
        output_device_index=1,
        save_file=None,
        quiet=False,
        logger=MagicMock(),
        play_audio=True,
        speed=1.0,
    )
    mock_speak_text.assert_called_once()
