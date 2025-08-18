"""Local HuggingFace model provider (placeholder)."""
from .base import LLMProvider


class HFLocal(LLMProvider):
    def generate(self, prompt: str) -> str:
        return "[hf placeholder]"
