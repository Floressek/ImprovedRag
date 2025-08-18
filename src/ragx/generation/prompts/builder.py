"""Jinja2 prompt templates (placeholder builder)."""
from typing import List


def build_answer_with_citations(question: str, contexts: List[str]) -> str:
    ctx = "\n\n".join(contexts)
    return f"[system]\nAnswer the question using the context. Cite sources.\n\n[context]\n{ctx}\n\n[question]\n{question}\n\n[answer]"
