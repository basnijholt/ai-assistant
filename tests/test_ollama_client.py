"""Tests for the Ollama client."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_cli.ollama_client import build_agent

if TYPE_CHECKING:
    import pytest


def test_build_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test building the Ollama agent."""
    monkeypatch.setenv("OLLAMA_HOST", "http://mockhost:1234")
    model = "test-model"
    host = "http://localhost:11434"

    agent = build_agent(model, host)

    assert agent.model.model_name == model
