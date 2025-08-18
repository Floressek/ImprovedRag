"""Retrieval-specific models (placeholder)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ScoredDoc:
    payload: Dict[str, Any]
    score: float
