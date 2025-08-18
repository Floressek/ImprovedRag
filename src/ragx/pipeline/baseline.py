"""Simple RAG pipeline (placeholder)."""
from typing import List, Tuple, Dict, Any

from ..generation.prompts.builder import build_answer_with_citations
from ..generation.inference import generate as llm_generate


def run(question: str, retrieved: List[Tuple[Dict[str, Any], float]]) -> str:
    contexts = [p.get("text", "") for p, _ in retrieved]
    prompt = build_answer_with_citations(question, contexts)
    return llm_generate(prompt)
