import json
import re
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


def safe_parse(text: str) -> Optional[Any]:
    """
    Parse JSON response safely. -> to be improved later

    Args:
        text: Response from LLM

    Returns:

    """
    if not text:
        logger.warning("Empty response from LLM")
        return None
    original_text = text
    text = text.strip()

    # markdown code block
    if text.startswith("```"):
        lines = text.split("\n")
        text = '\n'.join(lines[1:-1]) if len(lines) > 2 else text

    text = text.replace("```json", "").replace("```", "").strip()

    json_obj_match = re.search(r'\{.*\}', text, re.DOTALL)
    json_list_match = re.search(r'\[.*\]', text, re.DOTALL)

    if json_obj_match:
        text = json_obj_match.group(0)
    elif json_list_match:
        text = json_list_match.group(0)
    try:
        parsed = json.loads(text)
        logger.debug(f"Successfully parsed JSON: type={type(parsed)}")
        return parsed
    except json.JSONDecodeError as e:
        logger.warning(
            f"JSON parse failed: {e}\n"
            f"Original text: {original_text[:300]}\n"
            f"Cleaned text: {text[:300]}"
        )
        return None
