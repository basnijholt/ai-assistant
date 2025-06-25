"""Tests for the process_manager module."""

from __future__ import annotations

import os
import signal
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest

from ai_assistant import process_manager


@pytest.fixture
def temp_pid_dir(monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """Create a temporary directory for PID files during testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        monkeypatch.setattr(process_manager, "PID_DIR", temp_path)
        yield temp_path


def test_get_pid_file(temp_pid_dir: Path) -> None:
    """Test PID file path generation."""
    pid_file = process_manager.get_pid_file("test-process")
    assert pid_file == temp_pid_dir / "test-process.pid"
    assert temp_pid_dir.exists()  # Should create the directory


def test_write_and_read_pid_file() -> None:
    """Test writing and reading PID files."""
    process_name = "test-process"
    test_pid = 12345

    # Write PID
    process_manager.write_pid_file(process_name, test_pid)

    # Check file exists and contains correct PID
    pid_file = process_manager.get_pid_file(process_name)
    assert pid_file.exists()
    assert pid_file.read_text().strip() == str(test_pid)


def test_write_pid_file_current_process() -> None:
    """Test writing current process PID when no PID specified."""
    process_name = "test-process"

    process_manager.write_pid_file(process_name)

    pid_file = process_manager.get_pid_file(process_name)
    assert pid_file.exists()
    assert int(pid_file.read_text().strip()) == os.getpid()


def test_read_pid_file_nonexistent() -> None:
    """Test reading PID file that doesn't exist."""
    result = process_manager.read_pid_file("nonexistent")
    assert result is None


@patch("os.kill")
def test_read_pid_file_process_exists(mock_kill: MagicMock) -> None:
    """Test reading PID file when process exists."""
    process_name = "test-process"
    test_pid = 12345

    # Mock os.kill to not raise exception (process exists)
    mock_kill.return_value = None

    # Write and read PID
    process_manager.write_pid_file(process_name, test_pid)
    result = process_manager.read_pid_file(process_name)

    assert result == test_pid
    mock_kill.assert_called_once_with(test_pid, 0)


@patch("os.kill")
def test_read_pid_file_process_not_exists(mock_kill: MagicMock) -> None:
    """Test reading PID file when process doesn't exist."""
    process_name = "test-process"
    test_pid = 12345

    # Mock os.kill to raise ProcessLookupError (process doesn't exist)
    mock_kill.side_effect = ProcessLookupError()

    # Write PID
    process_manager.write_pid_file(process_name, test_pid)

    # Read should return None and clean up file
    result = process_manager.read_pid_file(process_name)

    assert result is None
    assert not process_manager.get_pid_file(process_name).exists()


def test_cleanup_pid_file() -> None:
    """Test cleaning up PID files."""
    process_name = "test-process"

    # Create PID file
    process_manager.write_pid_file(process_name, 12345)
    pid_file = process_manager.get_pid_file(process_name)
    assert pid_file.exists()

    # Cleanup
    process_manager.cleanup_pid_file(process_name)
    assert not pid_file.exists()


def test_cleanup_pid_file_nonexistent() -> None:
    """Test cleaning up PID file that doesn't exist (should not error)."""
    # Should not raise exception
    process_manager.cleanup_pid_file("nonexistent")


@patch("ai_assistant.process_manager.read_pid_file")
def test_is_process_running(mock_read_pid: MagicMock) -> None:
    """Test checking if process is running."""
    # Process running
    mock_read_pid.return_value = 12345
    assert process_manager.is_process_running("test") is True

    # Process not running
    mock_read_pid.return_value = None
    assert process_manager.is_process_running("test") is False


@patch("os.kill")
@patch("ai_assistant.process_manager.read_pid_file")
@patch("ai_assistant.process_manager.cleanup_pid_file")
def test_kill_process_success(
    mock_cleanup: MagicMock,
    mock_read_pid: MagicMock,
    mock_kill: MagicMock,
) -> None:
    """Test successfully killing a process."""
    mock_read_pid.return_value = 12345
    mock_kill.return_value = None

    result = process_manager.kill_process("test-process")

    assert result is True
    mock_kill.assert_called_once_with(12345, signal.SIGTERM)
    mock_cleanup.assert_called_once_with("test-process")


@patch("ai_assistant.process_manager.read_pid_file")
def test_kill_process_not_found(mock_read_pid: MagicMock) -> None:
    """Test killing process that doesn't exist."""
    mock_read_pid.return_value = None

    result = process_manager.kill_process("test-process")

    assert result is False


@patch("os.kill")
@patch("ai_assistant.process_manager.read_pid_file")
@patch("ai_assistant.process_manager.cleanup_pid_file")
def test_kill_process_already_dead(
    mock_cleanup: MagicMock,
    mock_read_pid: MagicMock,
    mock_kill: MagicMock,
) -> None:
    """Test killing process that's already dead."""
    mock_read_pid.return_value = 12345
    mock_kill.side_effect = ProcessLookupError()

    result = process_manager.kill_process("test-process")

    assert result is False
    mock_cleanup.assert_called_once_with("test-process")


def test_setup_signal_handlers() -> None:
    """Test signal handler setup."""
    callback = MagicMock()

    with patch("signal.signal") as mock_signal:
        process_manager.setup_signal_handlers("test-process", callback)

        # Should set up both SIGINT and SIGTERM handlers
        assert mock_signal.call_count == 2
        calls = mock_signal.call_args_list
        signals_set = {call[0][0] for call in calls}
        assert signals_set == {signal.SIGINT, signal.SIGTERM}


@patch("ai_assistant.process_manager.cleanup_pid_file")
@pytest.mark.skip(reason="Daemonization is not compatible with pytest's capture mechanism")
def test_daemonize_success(
    mock_cleanup: MagicMock,
    mock_setup_signal: MagicMock,  # noqa: ARG001
    mock_write_pid: MagicMock,
    mock_is_running: MagicMock,
) -> None:
    """Test successful daemonization."""
    mock_is_running.return_value = False

    main_function = MagicMock()

    with (
        patch("os.fork", return_value=0),
        patch("sys.exit"),
        patch("os.setsid"),
        patch("os.chdir"),
        patch("os.umask"),
    ):
        process_manager.daemonize("test-process", main_function)

    mock_write_pid.assert_called_once_with("test-process")
    main_function.assert_called_once()
    mock_cleanup.assert_called_once_with("test-process")


@patch("ai_assistant.process_manager.is_process_running")
@patch("ai_assistant.process_manager.read_pid_file")
def test_daemonize_already_running(
    mock_read_pid: MagicMock,
    mock_is_running: MagicMock,
) -> None:
    """Test daemonization when process is already running."""
    mock_is_running.return_value = True
    mock_read_pid.return_value = 12345

    main_function = MagicMock()

    with pytest.raises(SystemExit) as exc_info:
        process_manager.daemonize("test-process", main_function)

    assert exc_info.value.code == 1
    main_function.assert_not_called()
