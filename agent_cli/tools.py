"""Tool definitions for the interactive agent."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel, Field


class ReadFileTool(BaseModel):
    """A tool for reading the content of a file."""

    path: str = Field(..., description="The path to the file to read.")

    def run(self) -> str:
        """Read the content of a file."""
        try:
            return Path(self.path).read_text()
        except FileNotFoundError:
            return f"Error: File not found at {self.path}"
        except OSError as e:
            return f"Error reading file: {e}"


class ExecuteCodeTool(BaseModel):
    """A tool for executing a shell command."""

    code: str = Field(..., description="The shell command to execute.")

    def run(self) -> str:
        """Execute a shell command and return the output."""
        try:
            result = subprocess.run(
                self.code.split(),
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error executing code: {e.stderr}"
        except FileNotFoundError:
            return f"Error: Command not found: {self.code.split()[0]}"
