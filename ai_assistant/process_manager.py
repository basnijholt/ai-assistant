"""Process management utilities for AI Assistant tools."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# Default location for PID files
PID_DIR = Path.home() / ".cache" / "ai-assistant"


def get_pid_file(process_name: str) -> Path:
    """Get the path to the PID file for a given process name."""
    PID_DIR.mkdir(parents=True, exist_ok=True)
    return PID_DIR / f"{process_name}.pid"


def write_pid_file(process_name: str, pid: int | None = None) -> None:
    """Write the current process PID to a file."""
    if pid is None:
        pid = os.getpid()

    pid_file = get_pid_file(process_name)
    pid_file.write_text(str(pid))


def read_pid_file(process_name: str) -> int | None:
    """Read PID from file, return None if file doesn't exist or is invalid."""
    pid_file = get_pid_file(process_name)

    if not pid_file.exists():
        return None

    try:
        pid = int(pid_file.read_text().strip())
        # Check if process is still running
        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks if process exists
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        # Invalid PID or process doesn't exist
        cleanup_pid_file(process_name)
        return None


def cleanup_pid_file(process_name: str) -> None:
    """Remove the PID file."""
    pid_file = get_pid_file(process_name)
    pid_file.unlink(missing_ok=True)


def is_process_running(process_name: str) -> bool:
    """Check if a process with the given name is currently running."""
    return read_pid_file(process_name) is not None


def kill_process(process_name: str) -> bool:
    """Kill a process by name. Returns True if process was killed, False if not found."""
    pid = read_pid_file(process_name)
    if pid is None:
        return False

    try:
        os.kill(pid, signal.SIGTERM)
        cleanup_pid_file(process_name)
        return True
    except ProcessLookupError:
        # Process already dead
        cleanup_pid_file(process_name)
        return False
    except PermissionError:
        # No permission to kill process
        return False


def setup_signal_handlers(
    process_name: str,
    cleanup_callback: Callable[[], None] | None = None,
) -> None:
    """Set up signal handlers for graceful shutdown."""

    def signal_handler(signum: int, _frame: Any) -> None:
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        if cleanup_callback:
            cleanup_callback()
        cleanup_pid_file(process_name)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def daemonize(process_name: str, main_function: Callable[[], None]) -> None:
    """Run a function as a daemon process using a double-fork."""
    # Check if already running
    if is_process_running(process_name):
        existing_pid = read_pid_file(process_name)
        if existing_pid:
            print(f"Process {process_name} is already running (PID: {existing_pid})")
        else:
            print(f"Process {process_name} is already running.")
        sys.exit(1)

    # Fork to create daemon
    try:
        if os.fork() > 0:
            # Parent process, exit.
            sys.exit(0)
    except (OSError, AttributeError) as e:
        print(f"Fork #1 failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Decouple from parent environment
    os.chdir("/")
    os.setsid()
    os.umask(0)

    # Second fork
    try:
        if os.fork() > 0:
            # Parent process, exit.
            sys.exit(0)
    except OSError as e:
        print(f"Fork #2 failed: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Child process runs from here ---

    # Write the PID file
    write_pid_file(process_name)

    # Redirect standard file descriptors to /dev/null
    sys.stdout.flush()
    sys.stderr.flush()
    devnull_path = Path(os.devnull)
    with devnull_path.open("rb") as f_in, devnull_path.open("ab") as f_out:
        if sys.stdin.isatty():  # Only redirect if it's a real TTY
            os.dup2(f_in.fileno(), sys.stdin.fileno())
        os.dup2(f_out.fileno(), sys.stdout.fileno())
        os.dup2(f_out.fileno(), sys.stderr.fileno())

    # Set up signal handlers for graceful shutdown
    def cleanup_and_exit(_signum: int, _frame: Any) -> None:
        cleanup_pid_file(process_name)
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup_and_exit)
    signal.signal(signal.SIGINT, cleanup_and_exit)

    # Run the main function and ensure cleanup
    try:
        main_function()
    finally:
        cleanup_pid_file(process_name)
