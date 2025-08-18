"""Vector store factory (placeholder)."""
from __future__ import annotations
from typing import Literal

from .base import VectorStore
from .faiss_store import FaissStore
from .qdrant_store import QdrantStore


def get_vector_store(kind: Literal["faiss", "qdrant"] = "faiss") -> VectorStore:
    if kind == "qdrant":
        return QdrantStore()
    return FaissStore()
