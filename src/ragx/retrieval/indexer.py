"""Index building from processed data (placeholder)."""
from __future__ import annotations
from typing import List, Dict, Any


class Indexer:
    def __init__(self) -> None:
        self.vectors: List[List[float]] = []
        self.payloads: List[Dict[str, Any]] = []

    def add(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        self.vectors.extend(vectors)
        self.payloads.extend(payloads)

    def save(self, path: str) -> None:
        _ = path

    def load(self, path: str) -> None:
        _ = path
