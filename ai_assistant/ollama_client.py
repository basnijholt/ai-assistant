"""Client for interacting with Ollama."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


def build_agent(
    model: str,
    ollama_host: str,
    *,
    system_prompt: str | None = None,
    instructions: str | None = None,
) -> Agent:
    """Construct and return a PydanticAI agent configured for local Ollama."""
    ollama_provider = OpenAIProvider(base_url=f"{ollama_host}/v1")
    ollama_model = OpenAIModel(model_name=model, provider=ollama_provider)
    return Agent(
        model=ollama_model,
        system_prompt=system_prompt or (),
        instructions=instructions,
    )
