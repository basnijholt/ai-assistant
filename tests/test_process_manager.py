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
from daemon.pidfile import PIDLockFile

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
    assert temp_pid_dir.exists()


def test_get_log_file(temp_pid_dir: Path) -> None:
    """Test log file path generation."""
    log_file = process_manager.get_log_file("test-process")
    assert log_file == temp_pid_dir / "test-process.log"
    assert temp_pid_dir.exists()


def test_is_process_running() -> None:
    """Test checking if a process is running."""
    process_name = "test-process"
    pid_file = process_manager.get_pid_file(process_name)
    pid_lock = PIDLockFile(pid_file)

    # Case 1: Process is not running
    assert not process_manager.is_process_running(process_name)

    # Case 2: Process is running
    with pid_lock:
        assert process_manager.is_process_running(process_name)


def test_read_pid_file() -> None:
    """Test reading a PID from a PID file."""
    process_name = "test-process"
    pid_file = process_manager.get_pid_file(process_name)
    pid_lock = PIDLockFile(pid_file)

    # Case 1: No PID file
    assert process_manager.read_pid_file(process_name) is None

    # Case 2: PID file exists
    with pid_lock:
        assert process_manager.read_pid_file(process_name) == os.getpid()


@patch("os.kill")
def test_kill_process_success(mock_os_kill: MagicMock) -> None:
    """Test successfully killing a process."""
    process_name = "test-process"
    pid_file = process_manager.get_pid_file(process_name)
    pid_lock = PIDLockFile(pid_file)

    pid_lock.acquire()
    result = process_manager.kill_process(process_name)
    assert result is True
    mock_os_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)
    assert not pid_lock.is_locked()
    assert not pid_file.exists()


def test_kill_process_not_running() -> None:
    """Test killing a process that is not running."""
    result = process_manager.kill_process("test-process")
    assert result is False


@patch("os.kill", side_effect=ProcessLookupError)
def test_kill_process_already_dead(
    mock_os_kill: MagicMock,  # noqa: ARG001
) -> None:
    """Test killing a process that is already dead (stale PID file)."""
    process_name = "test-process"
    pid_file = process_manager.get_pid_file(process_name)
    pid_lock = PIDLockFile(pid_file)

    pid_lock.acquire()
    result = process_manager.kill_process(process_name)
    assert result is True
    assert not pid_lock.is_locked()
    assert not pid_file.exists()


@patch("daemon.DaemonContext")
@patch("daemon.pidfile.PIDLockFile")
def test_daemonize_success(
    mock_pid_lock_file: MagicMock,
    mock_daemon_context: MagicMock,
) -> None:
    """Test successful daemonization."""
    mock_lock = MagicMock()
    mock_pid_lock_file.return_value = mock_lock
    mock_lock.is_locked.return_value = False
    main_func = MagicMock()

    process_manager.daemonize("test-process", main_func)

    mock_daemon_context.assert_called_once()
    main_func.assert_called_once()


@patch("os.kill")
def test_daemonize_already_running(
    mock_os_kill: MagicMock,  # noqa: ARG001
) -> None:
    """Test daemonization when a process is already running."""
    process_name = "test-process"
    pid_file = process_manager.get_pid_file(process_name)
    pid_lock = PIDLockFile(pid_file)
    main_func = MagicMock()

    with pid_lock, pytest.raises(SystemExit) as e:
        process_manager.daemonize(process_name, main_func)

    assert e.value.code == 1
    main_func.assert_not_called()
