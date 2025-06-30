"""Tests for the TTS common module."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_cli.agents._tts_common import _save_audio_file, handle_tts_playback


@pytest.mark.asyncio
async def test_save_audio_file_os_error():
    """Test that an OS error is handled when saving an audio file."""
    mock_console = MagicMock()
    logger = logging.getLogger(__name__)
    with patch("pathlib.Path.write_bytes", side_effect=OSError("Test error")):
        await _save_audio_file(b"audio_data", Path("test.wav"), mock_console, logger)
        mock_console.print.assert_called()


@pytest.mark.asyncio
async def test_handle_tts_playback_os_error():
    """Test that an OS error is handled during TTS playback."""
    mock_console = MagicMock()
    logger = logging.getLogger(__name__)
    with patch("agent_cli.tts.speak_text", side_effect=OSError("Test error")):
        result = await handle_tts_playback(
            "text",
            tts_server_ip="localhost",
            tts_server_port=10200,
            voice_name=None,
            tts_language=None,
            speaker=None,
            output_device_index=None,
            save_file=None,
            console=mock_console,
            logger=logger,
        )
        assert result is None
        mock_console.print.assert_called()
