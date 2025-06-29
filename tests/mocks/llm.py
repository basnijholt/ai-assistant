"""Mock LLM agents and responses for testing."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class MockLLMResult:
    """Mock result from LLM agent execution."""

    def __init__(self, output: str, metadata: dict[str, Any] | None = None) -> None:
        """Initialize mock result."""
        self.data = output
        self.output = output
        self.metadata = metadata or {}


class MockLLMAgent:
    """Mock LLM agent for testing without real API calls."""

    def __init__(
        self,
        model: str,
        ollama_host: str,
        responses: dict[str, str],
        simulate_delay: float = 0.1,
    ) -> None:
        """Initialize mock agent.

        Args:
            model: Model name for the agent
            ollama_host: Ollama host URL
            responses: Mapping of input patterns to responses
            simulate_delay: Delay to simulate processing time

        """
        self.model = model
        self.ollama_host = ollama_host
        self.system_prompt = ""
        self.instructions = ""
        self.responses = responses
        self.simulate_delay = min(simulate_delay, 0.01)  # Cap at 10ms for tests
        self.call_history: list[dict[str, Any]] = []

    async def run_sync(
        self,
        user_prompt: str,
        *,
        model_dump: bool = True,
    ) -> MockLLMResult:
        """Mock synchronous execution of the agent."""
        # Record the call
        call_info = {
            "user_prompt": user_prompt,
            "model_dump": model_dump,
        }
        self.call_history.append(call_info)

        # Add small delay to simulate processing
        if self.simulate_delay > 0:
            await asyncio.sleep(self.simulate_delay)

        # Find matching response
        response = self._get_response_for_prompt(user_prompt)

        return MockLLMResult(response)

    def run(
        self,
        user_prompt: str,
        *,
        model_dump: bool = True,
    ) -> MockLLMResult:
        """Mock execution of the agent."""
        # Record the call
        call_info = {
            "user_prompt": user_prompt,
            "model_dump": model_dump,
        }
        self.call_history.append(call_info)

        # Find matching response
        response = self._get_response_for_prompt(user_prompt)

        return MockLLMResult(response)

    def _get_response_for_prompt(self, prompt: str) -> str:
        """Get appropriate response for the given prompt."""
        prompt_lower = prompt.lower()

        # Check for specific patterns
        for pattern, response in self.responses.items():
            if pattern.lower() in prompt_lower:
                return response

        # Default response
        return self.responses.get("default", "Mock LLM response")


def mock_build_agent(
    model: str,
    ollama_host: str,
    responses: dict[str, str],
    simulate_delay: float = 0.1,
) -> MockLLMAgent:
    """Create a mock agent that simulates the build_agent function."""
    return MockLLMAgent(
        model=model,
        ollama_host=ollama_host,
        responses=responses,
        simulate_delay=simulate_delay,
    )


def create_autocorrect_responses() -> dict[str, str]:
    """Create standard responses for autocorrect testing."""
    return {
        "hello": "Hello, world!",
        "test": "This is a corrected test message.",
        "default": "Text has been corrected.",
        "correct": "The text has been properly corrected with improved grammar and formatting.",
    }


def create_voice_assistant_responses() -> dict[str, str]:
    """Create standard responses for voice assistant testing."""
    return {
        "hello": "Hello! How can I help you today?",
        "question": "That's an interesting question. Here's my response.",
        "edit": "I've edited the text as requested.",
        "summarize": "Here's a summary of the text.",
        "translate": "Here's the translation you requested.",
        "default": "I understand your request and here is my response.",
    }


def create_conversation_responses() -> dict[str, str]:
    """Create responses for multi-turn conversation testing."""
    return {
        "greeting": "Hello! What would you like to do today?",
        "task": "I can help you with that task.",
        "follow_up": "Is there anything else you'd like me to help with?",
        "goodbye": "Goodbye! Have a great day!",
        "default": "I'm here to help with whatever you need.",
    }


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(
        self,
        *,
        base_url: str = "http://mock-llm:11434",
        responses: dict[str, str] | None = None,
    ) -> None:
        """Initialize mock provider.

        Args:
            base_url: Mock base URL
            responses: Response mappings

        """
        self.base_url = base_url
        self.responses = responses or {}


class MockLLMModel:
    """Mock LLM model for testing."""

    def __init__(
        self,
        *,
        model_name: str = "mock-model",
        provider: MockLLMProvider | None = None,
    ) -> None:
        """Initialize mock model.

        Args:
            model_name: Name of the mock model
            provider: Mock provider instance

        """
        self.model_name = model_name
        self.provider = provider or MockLLMProvider()
