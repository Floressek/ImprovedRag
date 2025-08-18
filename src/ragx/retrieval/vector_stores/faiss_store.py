"""FAISS fallback implementation (placeholder)."""
from __future__ import annotations
from typing import List, Tuple, Dict, Any


class FaissStore:
    def __init__(self) -> None:
        self.vecs: List[List[float]] = []
        self.payloads: List[Dict[str, Any]] = []

    def add(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        self.vecs.extend(vectors)
        self.payloads.extend(payloads)

    def search(self, vector: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        _ = vector
        return [(p, 0.0) for p in self.payloads[:top_k]]

    def save(self, path: str) -> None:
        _ = path

    def load(self, path: str) -> None:
        _ = path