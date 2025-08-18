"""Provider selection factory (placeholder)."""
from __future__ import annotations
from typing import Literal

from .base import LLMProvider
from .huggingface import HFLocal
from .openai_api import OpenAIClient


def get_provider(kind: Literal["hf", "openai"] = "hf") -> LLMProvider:
    if kind == "openai":
        return OpenAIClient()
    return HFLocal()
