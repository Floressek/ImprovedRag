"""Ingestion pipeline orchestration (placeholder)."""
from typing import List

from .wiki_extractor import extract_wiki
from .chunker import chunk_text


def run_ingestion(xml_path: str) -> List[str]:
    docs = extract_wiki(xml_path)
    chunks: List[str] = []
    for d in docs:
        chunks.extend(chunk_text(d))
    return chunks
