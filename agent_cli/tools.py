"""Tool definitions for the interactive agent."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic_ai.tools import Tool


def read_file(path: str) -> str:
    """Read the content of a file.

    Args:
        path: The path to the file to read.

    """
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return f"Error: File not found at {path}"
    except OSError as e:
        return f"Error reading file: {e}"


def execute_code(code: str) -> str:
    """Execute a shell command.

    Args:
        code: The shell command to execute.

    """
    try:
        result = subprocess.run(
            code.split(),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing code: {e.stderr}"
    except FileNotFoundError:
        return f"Error: Command not found: {code.split()[0]}"


ReadFileTool = Tool(read_file)
ExecuteCodeTool = Tool(execute_code)
