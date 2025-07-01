"""Example of tool calling with the agent CLI.

This example demonstrates how to use tools with the LLM agents.
"""

from __future__ import annotations

import asyncio
import logging

from agent_cli.llm import get_llm_response
from agent_cli.tools import ExecuteCodeTool, ReadFileTool
from agent_cli.utils import console

SYSTEM_PROMPT = """
You are a helpful assistant with access to tools.
"""

AGENT_INSTRUCTIONS = """
The user will ask a question. Your job is to answer the question.
"""

USER_INPUT = "What is in the file /tmp/test.txt?"


async def main() -> None:
    """Run the tool calling example."""
    logging.basicConfig(level="INFO")
    tools = [ReadFileTool, ExecuteCodeTool]
    console.print(f"[bold]User:[/bold] {USER_INPUT}")
    response = await get_llm_response(
        system_prompt=SYSTEM_PROMPT,
        agent_instructions=AGENT_INSTRUCTIONS,
        user_input=USER_INPUT,
        model="qwen3:4b",
        ollama_host="http://localhost:11434",
        logger=logging.getLogger(__name__),
        tools=tools,
    )
    console.print(f"[bold]Assistant:[/bold] {response}")
    console.print("[bold]User:[/bold] What is the current date?")
    response = await get_llm_response(
        system_prompt=SYSTEM_PROMPT,
        agent_instructions=AGENT_INSTRUCTIONS,
        user_input="What is the current date?",
        model="qwen3:4b",
        ollama_host="http://localhost:11434",
        logger=logging.getLogger(__name__),
        tools=tools,
    )
    console.print(f"[bold]Assistant:[/bold] {response}")


if __name__ == "__main__":
    asyncio.run(main())
