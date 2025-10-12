from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


def format_chat_history(
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_history: Optional[int] = None
) -> str:
    """Format chat history for a prompt.

    Args:
        chat_history: List of chat messages with 'role' and 'content'
        max_history: Maximum number of recent messages to include
    """
    if not chat_history:
        return ""

    max_history = max_history if max_history is not None else settings.chat.max_history

    if max_history is not None and len(chat_history) > max_history:
        # if the max history is exceeded, take the most recent messages - last in the list is most recent
        recent = chat_history[-max_history:]
    else:
        recent = chat_history

    formatted = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted.append(f"{role.capitalize()}: {content}")

    return "\n".join(formatted)


def build_rag_prompt(
        query: str,
        contexts: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_history: Optional[int] = None,
        system_instructions: Optional[str] = None,
) -> str:
    """Format chat history for a prompt.
    Args:
        query: The user's query string
        contexts: List of context documents with 'text' field
        chat_history: List of chat messages with 'role' and 'content'
        max_history: Maximum number of recent messages to include
        system_instructions: Custom system instructions to override default
    """
    default_system_prompt = """
    You are a helpful assistant. Answer questions using ONLY the provided context sources.
    Every factual claim MUST be cited using [N] format, where N is the source number.
    If the answer is not in the sources, say "I cannot find this information in the provided sources."
    Be concise and accurate.
    """

    system_prompt = system_instructions or default_system_prompt

    history_text = format_chat_history(chat_history, max_history)

    context_parts = []
    for idx, ctx in enumerate(contexts, 1):
        text = ctx.get("text", "")
        title = ctx.get("doc_title", "Unknown")  # add it from metadata from qdrant
        context_parts.append(f"[{idx}] (Source: {title})\n{text}")

    contexts_text = "\n\n".join(context_parts)

    prompt_parts = [
        "[SYSTEM]",
        system_prompt,
        "",
    ]

    if history_text:
        prompt_parts.extend([
            "[CHAT HISTORY]",
            history_text,
            "",
        ])

    prompt_parts.extend([
        "[CONTEXT]",
        contexts_text,
        "",
        "[QUESTION]",
        query,
        "",
        "[ANSWER]",
    ])

    return "\n".join(prompt_parts)
