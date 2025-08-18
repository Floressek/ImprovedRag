"""/ask endpoint handler (placeholder)."""
from __future__ import annotations
from typing import List, Dict, Any


def ask(question: str, contexts: List[str] | None = None) -> Dict[str, Any]:
    contexts = contexts or []
    return {"answer": "[placeholder]", "citations": [f"ctx:{i}" for i, _ in enumerate(contexts, 1)]}
