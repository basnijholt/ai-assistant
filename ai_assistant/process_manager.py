"""Process management utilities for AI Assistant tools."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import daemon
from daemon.pidfile import PIDLockFile

if TYPE_CHECKING:
    from collections.abc import Callable

# Default location for PID files
PID_DIR = Path.home() / ".cache" / "ai-assistant"


def get_pid_file(process_name: str) -> Path:
    """Get the path to the PID file for a given process name."""
    PID_DIR.mkdir(parents=True, exist_ok=True)
    return PID_DIR / f"{process_name}.pid"


def get_log_file(process_name: str) -> Path:
    """Get the path to the log file for a given process name."""
    PID_DIR.mkdir(parents=True, exist_ok=True)
    return PID_DIR / f"{process_name}.log"


def is_process_running(process_name: str) -> bool:
    """Check if a process with the given name is currently running."""
    pid_file_path = get_pid_file(process_name)
    pid_lock = PIDLockFile(pid_file_path)
    return pid_lock.is_locked()


def read_pid_file(process_name: str) -> int | None:
    """Read PID from file, return None if file doesn't exist or is invalid."""
    pid_file_path = get_pid_file(process_name)
    pid_lock = PIDLockFile(pid_file_path)

    if not pid_lock.is_locked():
        return None

    return pid_lock.read_pid()


def kill_process(process_name: str) -> bool:
    """Kill a process by name. Returns True if process was killed, False if not found."""
    pid_file_path = get_pid_file(process_name)
    pid_lock = PIDLockFile(pid_file_path)
    pid = pid_lock.read_pid()

    if not pid or not pid_lock.is_locked():
        return False

    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for the process to terminate
        for _ in range(10):  # Wait up to 1 second
            if not is_process_running(process_name):
                break
            time.sleep(0.1)
    except ProcessLookupError:
        # Process already dead
        pass
    except PermissionError:
        # No permission to kill process
        return False
    finally:
        # Clean up the lock file if it's still locked
        if pid_lock.is_locked():
            pid_lock.break_lock()

    return not is_process_running(process_name)


def daemonize(process_name: str, main_function: Callable[[], None]) -> None:
    """Run a function as a daemon process using python-daemon."""
    pid_file_path = get_pid_file(process_name)
    pid_lock = PIDLockFile(pid_file_path)

    if pid_lock.is_locked():
        existing_pid = pid_lock.read_pid()
        if existing_pid is not None:
            try:
                # Check if the process is actually running
                os.kill(existing_pid, 0)  # Signal 0 checks if process exists
                print(f"Process {process_name} is already running (PID: {existing_pid})")
                sys.exit(1)
            except (ProcessLookupError, PermissionError):
                # Process is dead, clean up stale lock
                print(f"Cleaning up stale PID lock for {process_name} (PID: {existing_pid})")
                pid_lock.break_lock()
        else:
            # PID file exists but is unreadable, clean it up
            print(f"Cleaning up corrupted PID lock for {process_name}")
            pid_lock.break_lock()

    log_file_path = get_log_file(process_name)

    with daemon.DaemonContext(
        pidfile=pid_lock,
        stdout=log_file_path.open("w"),
        stderr=log_file_path.open("w"),
        detach_process=True,  # Detach from the user session to run in the background
        prevent_core=True,
    ):
        main_function()
