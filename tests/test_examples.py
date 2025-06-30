"""Tests for the examples."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from examples import tool_calling


@pytest.mark.asyncio
@patch("examples.tool_calling.get_llm_response", new_callable=AsyncMock)
async def test_tool_calling_example(mock_get_llm_response: AsyncMock) -> None:
    """Test the tool_calling example."""
    mock_get_llm_response.return_value = "Mocked response"
    await tool_calling.main()
    assert mock_get_llm_response.call_count == 2
