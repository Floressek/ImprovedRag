"""SentenceTransformers wrapper (placeholder)."""
from typing import List

def embed_texts(texts: List[str]) -> List[List[float]]:
    return [[0.0, 0.0, 0.0] for _ in texts]


def embed_query(query: str) -> List[float]:
    return [0.0, 0.0, 0.0]
