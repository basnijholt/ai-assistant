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
        live=MagicMock(),
    )

    assert response == "hello"
    mock_build_agent.assert_called_once()
    mock_agent.run.assert_called_once_with("test")


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
        live=MagicMock(),
    )

    assert response is None
    mock_build_agent.assert_called_once()
    mock_agent.run.assert_called_once_with("test")


@patch("agent_cli.llm.get_llm_response", new_callable=AsyncMock)
def test_process_and_update_clipboard(
    mock_get_llm_response: AsyncMock,
) -> None:
    """Test the process_and_update_clipboard function."""
    mock_get_llm_response.return_value = "hello"
    mock_live = MagicMock()

    asyncio.run(
        process_and_update_clipboard(
            system_prompt="test",
            agent_instructions="test",
            model="test",
            ollama_host="test",
            logger=MagicMock(),
            original_text="test",
            instruction="test",
            clipboard=True,
            quiet=True,
            live=mock_live,
        ),
    )

    # Verify get_llm_response was called with the right parameters
    mock_get_llm_response.assert_called_once()
    call_args = mock_get_llm_response.call_args
    assert call_args.kwargs["clipboard"] is True
    assert call_args.kwargs["quiet"] is True
    assert call_args.kwargs["live"] is mock_live
    assert call_args.kwargs["show_output"] is True
    assert call_args.kwargs["exit_on_error"] is True
