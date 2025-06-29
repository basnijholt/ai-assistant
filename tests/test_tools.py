"""Tests for the tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_cli.tools import ExecuteCodeTool, ReadFileTool

if TYPE_CHECKING:
    from pathlib import Path


def test_read_file_tool(tmp_path: Path) -> None:
    """Test the ReadFileTool."""
    # 1. Test reading a file that exists
    file = tmp_path / "test.txt"
    file.write_text("hello")
    tool = ReadFileTool(path=str(file))
    assert tool.run() == "hello"

    # 2. Test reading a file that does not exist
    tool = ReadFileTool(path="non_existent_file.txt")
    assert "Error: File not found" in tool.run()


def test_execute_code_tool() -> None:
    """Test the ExecuteCodeTool."""
    # 1. Test a simple command
    tool = ExecuteCodeTool(code="echo hello")
    assert tool.run().strip() == "hello"

    # 2. Test a command that fails
    tool = ExecuteCodeTool(code="non_existent_command")
    assert "Error: Command not found" in tool.run()
