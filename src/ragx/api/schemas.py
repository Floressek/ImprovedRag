"""Pydantic-like request/response models (placeholders)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class AskRequest:
    question: str
    top_k: int = 5


@dataclass
class AskResponse:
    answer: str
    citations: List[str]


@dataclass
class SearchResponse:
    results: List[Dict[str, Any]]
