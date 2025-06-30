"""Examples of how to use the tool-calling feature."""

from __future__ import annotations

import asyncio
import logging

from agent_cli.llm import get_llm_response
from agent_cli.tools import ExecuteCodeTool, ReadFileTool


async def main() -> None:
    """Run the tool-calling examples."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    tools = [ReadFileTool, ExecuteCodeTool]
    user_input = "Read the content of the README.md file."
    response = await get_llm_response(
        system_prompt="You are a helpful assistant.",
        agent_instructions="Use the available tools to answer the user's question.",
        user_input=user_input,
        model="devstral:24b",
        ollama_host="http://localhost:11434",
        logger=logger,
        console=None,
        tools=tools,
    )
    print(f"Response for reading a file:\n{response}\n")

    user_input = "List the files in the current directory."
    response = await get_llm_response(
        system_prompt="You are a helpful assistant.",
        agent_instructions="Use the available tools to answer the user's question.",
        user_input=user_input,
        model="devstral:24b",
        ollama_host="http://localhost:11434",
        logger=logger,
        console=None,
        tools=tools,
    )
    print(f"Response for listing files:\n{response}\n")


if __name__ == "__main__":
    asyncio.run(main())
