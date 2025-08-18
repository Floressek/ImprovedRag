"""Pipeline selection and execution (placeholder)."""
from __future__ import annotations
from typing import List, Tuple, Dict, Any

from .baseline import run as run_baseline
from .enhanced import run as run_enhanced


def run(kind: str, question: str, retrieved: List[Tuple[Dict[str, Any], float]] | None = None) -> str:
    if kind == "enhanced":
        return run_enhanced(question, retrieved or [])
    return run_baseline(question, retrieved or [])
