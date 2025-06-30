"""Tests for the tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_cli.tools import execute_code, read_file

if TYPE_CHECKING:
    from pathlib import Path


def test_read_file_tool(tmp_path: Path) -> None:
    """Test the ReadFileTool."""
    # 1. Test reading a file that exists
    file = tmp_path / "test.txt"
    file.write_text("hello")
    assert read_file(path=str(file)) == "hello"

    # 2. Test reading a file that does not exist
    assert "Error: File not found" in read_file(path="non_existent_file.txt")


def test_execute_code_tool() -> None:
    """Test the ExecuteCodeTool."""
    # 1. Test a simple command
    assert execute_code(code="echo hello").strip() == "hello"

    # 2. Test a command that fails
    assert "Error: Command not found" in execute_code(code="non_existent_command")
