"""Text chunking with overlap (placeholder)."""
from typing import List

def chunk_text(text: str, size: int = 300, overlap: int = 50) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += max(1, size - overlap)
    return chunks
