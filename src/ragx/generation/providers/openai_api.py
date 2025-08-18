"""OpenAI API client provider (placeholder)."""
from .base import LLMProvider


class OpenAIClient(LLMProvider):
    def generate(self, prompt: str) -> str:
        return "[openai placeholder]"
