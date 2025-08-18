"""Text utilities (placeholder)."""
from typing import List

def sentence_split(text: str) -> List[str]:
    return [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]


def token_count(text: str) -> int:
    return len(text.split())
