from typing import List, Dict, Any


def get_quality_check_reminder(language: str) -> str:
    """
    Add final quality check reminder.

    Args:
        language: Language of the prompt
    """
    if language == "Polish":
        return """

        KOŃCOWA WERYFIKACJA:
        ✓ Każde twierdzenie ma cytat [N]
        ✓ Odpowiedź tylko ze źródeł
        ✓ Język polski w całości
        ✓ Sprzeczności oznaczone
        """
    else:
        return """
        
        FINAL VERIFICATION:
        ✓ Every claim has citation [N]
        ✓ Answer only from sources
        ✓ Consistent language throughout
        ✓ Contradictions noted
        """


def get_confidence_level(score: float) -> str:
    """
    Determine confidence level from score.

    Args:
        score: Confidence score
    """
    if score > 0.7:
        return "high"
    elif score > 0.4:
        return "medium"
    else:
        return "low"


def format_contexts_simple(
        contexts: List[Dict[str, Any]]
) -> str:
    """
    Format contexts for a basic template.

    Args:
        contexts: List of contexts
    """
    formatted = []
    for idx, ctx in enumerate(contexts, 1):
        text = ctx.get("text", "").strip()
        title = ctx.get("doc_title", "Unknown")
        formatted.append(f"[{idx}] {title}: {text}")
    return "\n\n".join(formatted)


def get_default_template() -> str:
    """Minimal fallback template."""
    return """
    Context: {contexts}
    Question: {query}
    Answer with citations [N]:
    """
