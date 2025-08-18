"""Core data models for ragx (placeholder)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Document:
    id: str
    text: str
    metadata: Optional[dict] = None


@dataclass
class Query:
    text: str
    top_k: int = 5


@dataclass
class Answer:
    text: str
    citations: Optional[List[str]] = None
