"""CoRAG activation rules (placeholder)."""

def should_use_corag(question: str) -> bool:
    return len(question.split()) > 12
