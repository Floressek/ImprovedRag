from typing import List


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Input text to split.

    Returns:
        List of sentences extracted from the input text.
    """
    import re

    # Prosty split na . ! ?
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]