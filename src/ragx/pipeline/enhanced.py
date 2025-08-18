"""Full pipeline with all enhancements (placeholder)."""
from __future__ import annotations
from typing import List, Tuple, Dict, Any

from .corag import multi_step_retrieval
from ..generation.prompts.builder import build_answer_with_citations
from ..generation.inference import generate as llm_generate
from ..verification.self_verify import self_verify


def run(question: str, retrieved: List[Tuple[Dict[str, Any], float]] | None = None) -> str:
    steps = multi_step_retrieval(question)
    contexts = [p.get("text", "") for p, _ in (retrieved or [])]
    prompt = build_answer_with_citations(question, contexts)
    answer = llm_generate(prompt)
    _ = self_verify(question, answer)
    return answer
