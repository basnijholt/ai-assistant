"""Tests for the Ollama client."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_cli.llm import (
    build_agent,
    get_llm_response,
    process_and_update_clipboard,
)


def test_build_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test building the Ollama agent."""
    monkeypatch.setenv("OLLAMA_HOST", "http://mockhost:1234")
    model = "test-model"
    host = "http://localhost:11434"

    agent = build_agent(model, host)

    assert agent.model.model_name == model


@pytest.mark.asyncio
@patch("agent_cli.llm.build_agent")
async def test_get_llm_response(mock_build_agent: MagicMock) -> None:
    """Test getting a response from the LLM."""
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=MagicMock(output="hello"))
    mock_build_agent.return_value = mock_agent

    response = await get_llm_response(
        system_prompt="test",
        agent_instructions="test",
        user_input="test",
        model="test",
        ollama_host="test",
        logger=MagicMock(),
        console=MagicMock(),
    )
    assert response == "hello"


@pytest.mark.asyncio
@patch("agent_cli.llm.build_agent")
async def test_get_llm_response_error(mock_build_agent: MagicMock) -> None:
    """Test getting a response from the LLM when an error occurs."""
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=Exception("test error"))
    mock_build_agent.return_value = mock_agent

    response = await get_llm_response(
        system_prompt="test",
        agent_instructions="test",
        user_input="test",
        model="test",
        ollama_host="test",
        logger=MagicMock(),
        console=MagicMock(),
    )
    assert response is None


@patch("agent_cli.llm.process_with_llm", new_callable=AsyncMock)
@patch("pyperclip.copy")
def test_process_and_update_clipboard(
    mock_copy: MagicMock,
    mock_process_with_llm: AsyncMock,
) -> None:
    """Test the process_and_update_clipboard function."""
    mock_process_with_llm.return_value = ("hello", 0.1)
    asyncio.run(
        process_and_update_clipboard(
            system_prompt="test",
            agent_instructions="test",
            model="test",
            ollama_host="test",
            logger=MagicMock(),
            console=MagicMock(),
            original_text="test",
            instruction="test",
            clipboard=True,
        ),
    )
    mock_copy.assert_called_once_with("hello")
