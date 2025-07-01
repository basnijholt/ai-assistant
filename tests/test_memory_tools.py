"""Tests for memory tool functions."""

import os
from pathlib import Path

from agent_cli._tools import (
    add_memory,
    list_all_memories,
    list_memory_categories,
    search_memory,
    update_memory,
)


def _setup_tmp_memory_dir(tmp_path: Path) -> None:
    """Point memory system at *tmp_path* for isolation."""
    os.environ["AGENT_CLI_HISTORY_DIR"] = str(tmp_path)


def test_add_and_search_memory(tmp_path: Path) -> None:
    _setup_tmp_memory_dir(tmp_path)

    # Add memory
    add_result = add_memory("User likes pizza", category="preferences", tags="food, pizza")
    assert "Memory added successfully" in add_result

    # Search memory
    search_result = search_memory("pizza")
    assert "likes pizza" in search_result
    assert "preferences" in search_result


def test_update_memory(tmp_path: Path) -> None:
    _setup_tmp_memory_dir(tmp_path)

    # Add initial memory
    add_memory("User lives in Paris", category="personal", tags="location, city")

    # Update memory ID 1
    update_result = update_memory(1, content="User lives in Berlin", tags="location, city, germany")
    assert "updated successfully" in update_result

    # Verify update via search
    search_result = search_memory("Berlin")
    assert "lives in Berlin" in search_result
    assert "germany" in search_result


def test_list_all_and_categories(tmp_path: Path) -> None:
    _setup_tmp_memory_dir(tmp_path)

    # Add multiple memories
    add_memory("User likes coffee", category="preferences", tags="drink, coffee")
    add_memory("Project deadline is next Friday", category="tasks", tags="deadline, project")

    # List categories
    categories = list_memory_categories()
    assert "preferences" in categories
    assert "tasks" in categories

    # List all memories
    all_memories = list_all_memories(limit=5)
    assert "likes coffee" in all_memories
    assert "deadline" in all_memories
