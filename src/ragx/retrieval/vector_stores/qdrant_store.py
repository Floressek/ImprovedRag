"""Qdrant implementation (placeholder)."""
from __future__ import annotations
from typing import List, Tuple, Dict, Any


class QdrantStore:
    def __init__(self) -> None:
        self._items: List[Dict[str, Any]] = []

    def add(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        # Placeholder: ignore vectors
        self._items.extend(payloads)

    def search(self, vector: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        _ = vector
        return [(p, 0.0) for p in self._items[:top_k]]

    def save(self, path: str) -> None:
        _ = path

    def load(self, path: str) -> None:
        _ = path
