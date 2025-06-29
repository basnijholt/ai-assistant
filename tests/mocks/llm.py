"""Mock LLM agents and responses for testing."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from collections.abc import Any


class MockLLMResult:
    """Mock result from LLM agent."""
    
    def __init__(self, output: str, *, metadata: dict[str, Any] | None = None) -> None:
        """Initialize mock LLM result.
        
        Args:
            output: The response text
            metadata: Optional metadata about the response
            
        """
        self.output = output
        self.metadata = metadata or {}


class MockLLMAgent:
    """Mock PydanticAI Agent for testing."""
    
    def __init__(
        self,
        *,
        responses: dict[str, str] | None = None,
        default_response: str = "Mock LLM response",
        simulate_delay: float = 0.1,
        model_name: str = "mock-model",
    ) -> None:
        """Initialize mock LLM agent.
        
        Args:
            responses: Mapping of input patterns to responses
            default_response: Default response for unmatched inputs
            simulate_delay: Delay to simulate processing time
            model_name: Name of the mock model
            
        """
        self.responses = responses or {}
        self.default_response = default_response
        self.simulate_delay = simulate_delay
        self.model = MagicMock()
        self.model.model_name = model_name
        self.call_history: list[str] = []
    
    async def run(self, user_input: str) -> MockLLMResult:
        """Run the mock agent with user input.
        
        Args:
            user_input: Input text to process
            
        Returns:
            Mock LLM result
            
        """
        self.call_history.append(user_input)
        
        # Simulate processing delay
        if self.simulate_delay > 0:
            await asyncio.sleep(self.simulate_delay)
        
        # Find matching response
        response = self._find_response(user_input)
        
        return MockLLMResult(
            output=response,
            metadata={
                "model": self.model.model_name,
                "input_length": len(user_input),
                "timestamp": time.time(),
            },
        )
    
    def _find_response(self, user_input: str) -> str:
        """Find appropriate response for the input."""
        input_lower = user_input.lower()
        
        # Check for exact pattern matches first
        for pattern, response in self.responses.items():
            if pattern.lower() in input_lower:
                return response
        
        # Check for common instruction patterns
        if "correct" in input_lower or "fix" in input_lower:
            return self.responses.get("correct", "Corrected text here.")
        
        if "question" in input_lower or "?" in user_input:
            return self.responses.get("question", "Here is my answer to your question.")
        
        if "hello" in input_lower or "hi" in input_lower:
            return self.responses.get("hello", "Hello! How can I help you?")
        
        return self.default_response


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


def mock_build_agent(
    model: str,
    ollama_host: str,
    *,
    system_prompt: str | None = None,
    instructions: str | None = None,
    responses: dict[str, str] | None = None,
    simulate_delay: float = 0.1,
) -> MockLLMAgent:
    """Create a mock LLM agent that mimics build_agent behavior.
    
    Args:
        model: Model name
        ollama_host: Ollama host URL
        system_prompt: System prompt (stored but not used in mock)
        instructions: Instructions (stored but not used in mock)
        responses: Custom response mappings
        simulate_delay: Processing delay simulation
        
    Returns:
        Mock LLM agent
        
    """
    agent = MockLLMAgent(
        responses=responses,
        simulate_delay=simulate_delay,
        model_name=model,
    )
    
    # Store prompt information for verification
    agent.system_prompt = system_prompt
    agent.instructions = instructions
    agent.ollama_host = ollama_host
    
    return agent


def create_autocorrect_responses() -> dict[str, str]:
    """Create typical autocorrect LLM responses."""
    return {
        "hello world": "Hello, world!",
        "this is a test": "This is a test.",
        "how are you": "How are you?",
        "i am fine": "I am fine.",
        "correct": "The text has been corrected with proper grammar and punctuation.",
        "fix": "The text has been fixed and improved.",
    }


def create_voice_assistant_responses() -> dict[str, str]:
    """Create typical voice assistant LLM responses."""
    return {
        "hello": "Hello! How can I help you today?",
        "how are you": "I'm doing well, thank you for asking! How are you?",
        "what time": "I don't have access to the current time, but I can help with other questions.",
        "weather": "I don't have access to weather information, but I can help with other topics.",
        "help": "I'm here to help! What would you like to know?",
        "goodbye": "Goodbye! It was nice talking with you.",
        "question": "That's a great question! Let me help you with that.",
    }


def create_conversation_responses() -> dict[str, str]:
    """Create realistic conversation responses for testing."""
    return {
        "What is the capital of France?": "The capital of France is Paris.",
        "How do I make coffee?": "To make coffee, you can use a coffee maker, French press, or pour-over method. Start with good quality coffee beans and water.",
        "Tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
        "What's the weather like?": "I don't have access to current weather data, but you can check a weather app or website for accurate information.",
        "How are you?": "I'm doing well, thank you for asking! I'm here to help with any questions you might have.",
        "What can you do?": "I can help with a variety of tasks including answering questions, providing information, and having conversations. What would you like to know?",
    } 