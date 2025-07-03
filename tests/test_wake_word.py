"""Tests for the wake word detection module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.live import Live

from agent_cli import wake_word
from agent_cli.utils import InteractiveStopEvent


@pytest.fixture
def mock_pyaudio():
    """Mock PyAudio instance."""
    return MagicMock()


@pytest.fixture
def mock_logger():
    """Mock logger instance."""
    return MagicMock()


@pytest.fixture
def mock_stop_event():
    """Mock stop event."""
    stop_event = MagicMock(spec=InteractiveStopEvent)
    stop_event.is_set.return_value = False
    stop_event.ctrl_c_pressed = False
    return stop_event


@pytest.fixture
def mock_live():
    """Mock Rich Live instance."""
    return MagicMock(spec=Live)


class TestSendAudioForWakeDetection:
    """Tests for send_audio_for_wake_detection function."""

    @pytest.mark.asyncio
    async def test_sends_audio_start_event(self, mock_logger, mock_stop_event, mock_live):
        """Test that audio start event is sent."""
        mock_client = AsyncMock()
        mock_stream = MagicMock()
        
        # Setup stop event to stop immediately
        mock_stop_event.is_set.side_effect = [True]
        
        await wake_word.send_audio_for_wake_detection(
            mock_client,
            mock_stream,
            mock_stop_event,
            mock_logger,
            live=mock_live,
            quiet=True,
        )
        
        # Verify AudioStart event was sent
        mock_client.write_event.assert_called()
        call_args = mock_client.write_event.call_args_list[0][0][0]
        assert call_args.type == "audio-start"

    @pytest.mark.asyncio
    async def test_sends_audio_stop_event_on_exit(self, mock_logger, mock_stop_event, mock_live):
        """Test that audio stop event is sent when exiting."""
        mock_client = AsyncMock()
        mock_stream = MagicMock()
        
        # Setup stop event to stop immediately
        mock_stop_event.is_set.side_effect = [True]
        
        await wake_word.send_audio_for_wake_detection(
            mock_client,
            mock_stream,
            mock_stop_event,
            mock_logger,
            live=mock_live,
            quiet=True,
        )
        
        # Verify AudioStop event was sent
        call_args_list = [call[0][0] for call in mock_client.write_event.call_args_list]
        audio_stop_events = [event for event in call_args_list if event.type == "audio-stop"]
        assert len(audio_stop_events) == 1

    @pytest.mark.asyncio
    @patch("agent_cli.wake_word.asyncio.to_thread")
    async def test_reads_and_sends_audio_chunks(self, mock_to_thread, mock_logger, mock_stop_event, mock_live):
        """Test that audio chunks are read and sent."""
        mock_client = AsyncMock()
        mock_stream = MagicMock()
        
        # Setup audio chunk data
        test_chunk = b"test_audio_data"
        mock_to_thread.return_value = test_chunk
        
        # Setup stop event to run once then stop
        mock_stop_event.is_set.side_effect = [False, True]
        
        await wake_word.send_audio_for_wake_detection(
            mock_client,
            mock_stream,
            mock_stop_event,
            mock_logger,
            live=mock_live,
            quiet=True,
        )
        
        # Verify audio chunk was sent
        call_args_list = [call[0][0] for call in mock_client.write_event.call_args_list]
        audio_chunk_events = [event for event in call_args_list if event.type == "audio-chunk"]
        assert len(audio_chunk_events) == 1
        assert audio_chunk_events[0].audio == test_chunk

    @pytest.mark.asyncio
    async def test_updates_live_display_with_timing(self, mock_logger, mock_stop_event, mock_live):
        """Test that live display is updated with timing information."""
        mock_client = AsyncMock()
        mock_stream = MagicMock()
        
        # Setup stop event to stop immediately
        mock_stop_event.is_set.side_effect = [True]
        
        await wake_word.send_audio_for_wake_detection(
            mock_client,
            mock_stream,
            mock_stop_event,
            mock_logger,
            live=mock_live,
            quiet=False,  # Not quiet to enable live updates
        )
        
        # With immediate stop, no live updates should occur
        # This test mainly ensures the function doesn't crash with live updates


class TestReceiveWakeDetection:
    """Tests for receive_wake_detection function."""

    @pytest.mark.asyncio
    async def test_returns_detected_wake_word(self, mock_logger):
        """Test detection of wake word."""
        mock_client = AsyncMock()
        
        # Mock detection event
        mock_event = MagicMock()
        mock_event.type = "detection"
        
        # Mock Detection.is_type and Detection.from_event
        with (
            patch("agent_cli.wake_word.Detection.is_type", return_value=True),
            patch("agent_cli.wake_word.Detection.from_event") as mock_from_event,
        ):
            mock_detection = MagicMock()
            mock_detection.name = "test_wake_word"
            mock_from_event.return_value = mock_detection
            
            mock_client.read_event.return_value = mock_event
            
            result = await wake_word.receive_wake_detection(mock_client, mock_logger)
            
            assert result == "test_wake_word"
            mock_logger.info.assert_called_with("Wake word detected: %s", "test_wake_word")

    @pytest.mark.asyncio
    async def test_calls_detection_callback(self, mock_logger):
        """Test that detection callback is called."""
        mock_client = AsyncMock()
        mock_callback = MagicMock()
        
        # Mock detection event
        mock_event = MagicMock()
        mock_event.type = "detection"
        
        with (
            patch("agent_cli.wake_word.Detection.is_type", return_value=True),
            patch("agent_cli.wake_word.Detection.from_event") as mock_from_event,
        ):
            mock_detection = MagicMock()
            mock_detection.name = "test_wake_word"
            mock_from_event.return_value = mock_detection
            
            mock_client.read_event.return_value = mock_event
            
            result = await wake_word.receive_wake_detection(
                mock_client, 
                mock_logger,
                detection_callback=mock_callback
            )
            
            assert result == "test_wake_word"
            mock_callback.assert_called_once_with("test_wake_word")

    @pytest.mark.asyncio
    async def test_handles_not_detected_event(self, mock_logger):
        """Test handling of not-detected event."""
        mock_client = AsyncMock()
        
        # Mock not-detected event
        mock_event = MagicMock()
        mock_event.type = "not-detected"
        
        with (
            patch("agent_cli.wake_word.Detection.is_type", return_value=False),
            patch("agent_cli.wake_word.NotDetected.is_type", return_value=True),
        ):
            mock_client.read_event.return_value = mock_event
            
            result = await wake_word.receive_wake_detection(mock_client, mock_logger)
            
            assert result is None
            mock_logger.debug.assert_called_with("No wake word detected")

    @pytest.mark.asyncio
    async def test_handles_connection_loss(self, mock_logger):
        """Test handling of lost connection."""
        mock_client = AsyncMock()
        mock_client.read_event.return_value = None
        
        result = await wake_word.receive_wake_detection(mock_client, mock_logger)
        
        assert result is None
        mock_logger.warning.assert_called_with("Connection to wake word server lost.")


class TestDetectWakeWord:
    """Tests for detect_wake_word function."""

    @pytest.mark.asyncio
    @patch("agent_cli.wake_word.AsyncClient.from_uri")
    @patch("agent_cli.wake_word.open_pyaudio_stream")
    async def test_successful_wake_word_detection(
        self, mock_stream_context, mock_client_from_uri, mock_pyaudio, mock_logger, mock_stop_event, mock_live
    ):
        """Test successful wake word detection."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_client_from_uri.return_value.__aenter__.return_value = mock_client
        
        mock_stream = MagicMock()
        mock_stream_context.return_value.__enter__.return_value = mock_stream
        
        # Mock the tasks to complete successfully
        with (
            patch("agent_cli.wake_word.send_audio_for_wake_detection") as mock_send,
            patch("agent_cli.wake_word.receive_wake_detection") as mock_receive,
            patch("asyncio.wait") as mock_wait,
        ):
            # Setup task mocks
            mock_send_task = AsyncMock()
            mock_receive_task = AsyncMock()
            mock_receive_task.result.return_value = "detected_word"
            
            mock_wait.return_value = ([mock_receive_task], [mock_send_task])
            
            result = await wake_word.detect_wake_word(
                wake_server_ip="127.0.0.1",
                wake_server_port=10400,
                wake_word_name="test_word",
                input_device_index=1,
                logger=mock_logger,
                p=mock_pyaudio,
                stop_event=mock_stop_event,
                live=mock_live,
                quiet=True,
            )
            
            assert result == "detected_word"
            # Verify detect event was sent
            mock_client.write_event.assert_called_once()

    @pytest.mark.asyncio
    @patch("agent_cli.wake_word.AsyncClient.from_uri")
    async def test_connection_refused_error(self, mock_client_from_uri, mock_pyaudio, mock_logger, mock_stop_event, mock_live):
        """Test handling of connection refused error."""
        mock_client_from_uri.side_effect = ConnectionRefusedError()
        
        with patch("agent_cli.wake_word.print_error_message") as mock_print_error:
            result = await wake_word.detect_wake_word(
                wake_server_ip="127.0.0.1",
                wake_server_port=10400,
                wake_word_name="test_word",
                input_device_index=1,
                logger=mock_logger,
                p=mock_pyaudio,
                stop_event=mock_stop_event,
                live=mock_live,
                quiet=False,  # Not quiet to test error message
            )
            
            assert result is None
            mock_print_error.assert_called_once()

    @pytest.mark.asyncio
    @patch("agent_cli.wake_word.AsyncClient.from_uri")
    async def test_generic_exception_handling(self, mock_client_from_uri, mock_pyaudio, mock_logger, mock_stop_event, mock_live):
        """Test handling of generic exceptions."""
        mock_client_from_uri.side_effect = Exception("Test error")
        
        with patch("agent_cli.wake_word.print_error_message") as mock_print_error:
            result = await wake_word.detect_wake_word(
                wake_server_ip="127.0.0.1",
                wake_server_port=10400,
                wake_word_name="test_word",
                input_device_index=1,
                logger=mock_logger,
                p=mock_pyaudio,
                stop_event=mock_stop_event,
                live=mock_live,
                quiet=False,
            )
            
            assert result is None
            mock_print_error.assert_called_once()
            mock_logger.exception.assert_called_once()

    @pytest.mark.asyncio
    @patch("agent_cli.wake_word.AsyncClient.from_uri")
    @patch("agent_cli.wake_word.open_pyaudio_stream")
    async def test_task_cancellation(
        self, mock_stream_context, mock_client_from_uri, mock_pyaudio, mock_logger, mock_stop_event, mock_live
    ):
        """Test that pending tasks are cancelled."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_client_from_uri.return_value.__aenter__.return_value = mock_client
        
        mock_stream = MagicMock()
        mock_stream_context.return_value.__enter__.return_value = mock_stream
        
        with (
            patch("agent_cli.wake_word.send_audio_for_wake_detection") as mock_send,
            patch("agent_cli.wake_word.receive_wake_detection") as mock_receive,
            patch("asyncio.wait") as mock_wait,
        ):
            # Setup task mocks - send task completes first, receive is pending
            mock_send_task = AsyncMock()
            mock_receive_task = AsyncMock()
            mock_receive_task.cancel = MagicMock()
            
            mock_wait.return_value = ([mock_send_task], [mock_receive_task])
            
            result = await wake_word.detect_wake_word(
                wake_server_ip="127.0.0.1",
                wake_server_port=10400,
                wake_word_name="test_word",
                input_device_index=1,
                logger=mock_logger,
                p=mock_pyaudio,
                stop_event=mock_stop_event,
                live=mock_live,
                quiet=True,
            )
            
            # Verify pending task was cancelled
            mock_receive_task.cancel.assert_called_once()
            assert result is None  # No result since receive task didn't complete