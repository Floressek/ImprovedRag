"""/search endpoint handler (placeholder)."""
from __future__ import annotations
from typing import List, Dict, Any


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return [{"text": f"result {i}", "score": 0.0} for i in range(top_k)]
