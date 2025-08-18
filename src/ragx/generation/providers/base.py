"""LLM provider interface (placeholder)."""
from __future__ import annotations
from typing import Protocol


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str: ...
