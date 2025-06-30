"""Client for interacting with Ollama."""

from __future__ import annotations

import sys
import time
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

import pyperclip
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from rich.status import Status

from agent_cli.utils import print_error_message, print_output_panel

if TYPE_CHECKING:
    import logging

    from pydantic_ai.tools import Tool
    from rich.console import Console


def build_agent(
    model: str,
    ollama_host: str,
    *,
    system_prompt: str | None = None,
    instructions: str | None = None,
    tools: list[Tool] | None = None,
) -> Agent:
    """Construct and return a PydanticAI agent configured for local Ollama."""
    ollama_provider = OpenAIProvider(base_url=f"{ollama_host}/v1")
    ollama_model = OpenAIModel(model_name=model, provider=ollama_provider)
    return Agent(
        model=ollama_model,
        system_prompt=system_prompt or (),
        instructions=instructions,
        tools=tools or [],
        model_settings=OpenAIResponsesModelSettings(extra_body={"think": False}),
    )


# --- LLM (Editing) Logic ---

INPUT_TEMPLATE = """
<original-text>
{original_text}
</original-text>

<instruction>
{instruction}
</instruction>
"""


async def get_llm_response(
    *,
    system_prompt: str,
    agent_instructions: str,
    user_input: str,
    model: str,
    ollama_host: str,
    logger: logging.Logger,
    console: Console | None,
    tools: list[Tool] | None = None,
) -> str | None:
    """Get a response from the LLM."""
    agent = build_agent(
        model=model,
        ollama_host=ollama_host,
        system_prompt=system_prompt,
        instructions=agent_instructions,
        tools=tools,
    )
    try:
        with _maybe_status(console, model):
            result = await agent.run(user_input)
        return result.output
    except Exception as e:
        logger.exception("An error occurred during LLM processing.")
        print_error_message(
            console,
            f"An unexpected LLM error occurred: {e}",
            f"Please check your Ollama server at [cyan]{ollama_host}[/cyan]",
        )
        return None


async def process_with_llm(
    agent: Agent,
    original_text: str,
    instruction: str,
) -> tuple[str, float]:
    """Run the agent asynchronously and return corrected text and elapsed time."""
    user_input = INPUT_TEMPLATE.format(
        original_text=original_text,
        instruction=instruction,
    )
    t_start = time.monotonic()
    result = await agent.run(user_input)
    t_end = time.monotonic()
    return result.output, t_end - t_start


def _maybe_status(
    console: Console | None,
    model: str,
) -> AbstractContextManager[Status | None]:
    """Context manager for status display."""
    if console:
        return Status(
            f"[bold yellow]🤖 Applying instruction with {model}...[/bold yellow]",
            console=console,
        )
    return nullcontext()


async def process_and_update_clipboard(
    system_prompt: str,
    agent_instructions: str,
    *,
    model: str,
    ollama_host: str,
    logger: logging.Logger,
    console: Console | None,
    original_text: str,
    instruction: str,
    clipboard: bool,
) -> None:
    """Processes the text with the LLM, updates the clipboard, and displays the result.

    In quiet mode, only the result is printed to stdout.
    """
    agent = build_agent(
        model=model,
        ollama_host=ollama_host,
        system_prompt=system_prompt,
        instructions=agent_instructions,
    )
    try:
        with _maybe_status(console, model):
            result_text, elapsed = await process_with_llm(
                agent,
                original_text,
                instruction,
            )

        if clipboard:
            pyperclip.copy(result_text)
            logger.info("Copied result to clipboard.")

        if console:
            print_output_panel(
                console,
                result_text,
                title="✨ Result (Copied to Clipboard)" if clipboard else "✨ Result",
                subtitle=f"[dim]took {elapsed:.2f}s[/dim]",
            )
        else:
            # Quiet mode: print result to stdout for Keyboard Maestro to capture
            print(result_text)

    except Exception as e:
        logger.exception("An error occurred during LLM processing.")
        print_error_message(
            console,
            f"An unexpected LLM error occurred: {e}",
            f"Please check your Ollama server at [cyan]{ollama_host}[/cyan]",
        )
        sys.exit(1)
