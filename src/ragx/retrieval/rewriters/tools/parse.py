import json
import re
import time
from typing import Optional, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry logic with exponential backoff."""
    max_retries: int = 3
    initial_delay: float = 0.5
    max_delay: float = 8.0
    exponential_base: float = 2.0
    jitter: bool = True


class JSONValidator:
    """Validates JSON responses with retry and exponential backoff."""

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize JSONValidator.

        Args:
            config: RetryConfig instance with retry parameters
        """
        self.config = config or RetryConfig()

    def validate_and_parse(
        self,
        text: str,
        parse_func: Optional[Callable[[str], Optional[Any]]] = None,
    ) -> tuple[bool, Optional[Any], Optional[str]]:
        """
        Validate and parse JSON text.

        Args:
            text: Text to parse as JSON
            parse_func: Optional custom parsing function (default: safe_parse)

        Returns:
            Tuple of (is_valid, parsed_data, error_message)
        """
        if not text or not text.strip():
            return False, None, "Empty response"

        parse_function = parse_func or safe_parse

        try:
            parsed = parse_function(text)
            if parsed is None:
                return False, None, "Parser returned None"
            return True, parsed, None
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {str(e)}"
            logger.debug(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Unexpected parsing error: {str(e)}"
            logger.warning(error_msg)
            return False, None, error_msg

    def validate_with_retry(
        self,
        generator_func: Callable[[], str],
        parse_func: Optional[Callable[[str], Optional[Any]]] = None,
        validator_func: Optional[Callable[[Any], bool]] = None,
    ) -> tuple[bool, Optional[Any], dict]:
        """
        Execute generator function with retry and exponential backoff.

        Args:
            generator_func: Function that generates JSON response (e.g., LLM call)
            parse_func: Optional custom parsing function
            validator_func: Optional function to validate parsed structure

        Returns:
            Tuple of (success, parsed_data, metadata)
            metadata contains: attempts, total_time, errors, last_delay
        """
        metadata = {
            "attempts": 0,
            "total_time": 0.0,
            "errors": [],
            "last_delay": 0.0,
        }

        start_time = time.time()
        delay = self.config.initial_delay

        for attempt in range(self.config.max_retries):
            metadata["attempts"] = attempt + 1

            try:
                # Generate response
                logger.debug(f"Attempt {attempt + 1}/{self.config.max_retries}")
                response = generator_func()

                # Validate and parse
                is_valid, parsed, error = self.validate_and_parse(response, parse_func)

                if is_valid and parsed is not None:
                    # Additional structural validation if provided
                    if validator_func and not validator_func(parsed):
                        error = "Structural validation failed"
                        is_valid = False
                        logger.warning(f"Attempt {attempt + 1}: {error}")
                        metadata["errors"].append({
                            "attempt": attempt + 1,
                            "error": error,
                            "response_preview": response[:200] if response else None,
                        })
                    else:
                        # Success!
                        metadata["total_time"] = time.time() - start_time
                        logger.info(
                            f"Successfully parsed JSON after {attempt + 1} attempt(s) "
                            f"in {metadata['total_time']:.2f}s"
                        )
                        return True, parsed, metadata
                else:
                    # Parse failed
                    logger.warning(f"Attempt {attempt + 1}: {error}")
                    metadata["errors"].append({
                        "attempt": attempt + 1,
                        "error": error,
                        "response_preview": response[:200] if response else None,
                    })

            except Exception as e:
                error_msg = f"Generator function error: {str(e)}"
                logger.error(f"Attempt {attempt + 1}: {error_msg}")
                metadata["errors"].append({
                    "attempt": attempt + 1,
                    "error": error_msg,
                })

            # Exponential backoff before retry (skip on last attempt)
            if attempt < self.config.max_retries - 1:
                # Calculate delay with exponential backoff
                delay = min(
                    self.config.initial_delay * (self.config.exponential_base ** attempt),
                    self.config.max_delay
                )

                # Add jitter if enabled
                if self.config.jitter:
                    import random
                    delay = delay * (0.5 + random.random())

                metadata["last_delay"] = delay
                logger.info(f"Retrying in {delay:.2f}s...")
                time.sleep(delay)

        # All retries exhausted
        metadata["total_time"] = time.time() - start_time
        logger.error(
            f"Failed to parse valid JSON after {self.config.max_retries} attempts "
            f"in {metadata['total_time']:.2f}s"
        )
        return False, None, metadata


def validate_rewriter_response(parsed: Any) -> bool:
    """
    Validate the structure of a rewriter response.

    Args:
        parsed: Parsed JSON object

    Returns:
        True if structure is valid, False otherwise
    """
    if not isinstance(parsed, dict):
        logger.warning(f"Expected dict, got {type(parsed)}")
        return False

    # Required fields
    required_fields = ["is_multihop", "action", "reasoning"]
    for field in required_fields:
        if field not in parsed:
            logger.warning(f"Missing required field: {field}")
            return False

    # Validate action field
    valid_actions = ["decompose", "expand", "passthrough"]
    if parsed["action"] not in valid_actions:
        logger.warning(f"Invalid action: {parsed['action']}, expected one of {valid_actions}")
        return False

    # Validate is_multihop is boolean
    if not isinstance(parsed["is_multihop"], bool):
        logger.warning(f"is_multihop must be boolean, got {type(parsed['is_multihop'])}")
        return False

    # Validate query_type if present (optional but should be valid if present)
    if "query_type" in parsed:
        valid_query_types = [
            "verification", "comparison", "similarity", "chaining",
            "temporal", "aggregation", "superlative", "simple"
        ]
        if parsed["query_type"] not in valid_query_types:
            logger.warning(
                f"Invalid query_type: {parsed['query_type']}, "
                f"expected one of {valid_query_types}"
            )
            return False

    # Action-specific validation
    if parsed["action"] == "decompose":
        if "sub_queries" not in parsed or not isinstance(parsed.get("sub_queries"), list):
            logger.warning("decompose action requires 'sub_queries' list")
            return False
        if not parsed["sub_queries"]:
            logger.warning("sub_queries list is empty")
            return False

    if parsed["action"] == "expand":
        if "expanded_query" not in parsed or not isinstance(parsed.get("expanded_query"), str):
            logger.warning("expand action requires 'expanded_query' string")
            return False

    # Validate confidence if present
    if "confidence" in parsed:
        confidence = parsed["confidence"]
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            logger.warning(f"Invalid confidence value: {confidence}")
            return False

    return True


def validate_verification_response(parsed: Any) -> bool:
    """
    Validate the structure of a verification response.

    Args:
        parsed: Parsed JSON object

    Returns:
        True if structure is valid, False otherwise
    """
    if not isinstance(parsed, dict):
        logger.warning(f"Expected dict, got {type(parsed)}")
        return False

    # Required fields
    required_fields = ["valid", "issues"]
    for field in required_fields:
        if field not in parsed:
            logger.warning(f"Missing required field: {field}")
            return False

    # Validate types
    if not isinstance(parsed["valid"], bool):
        logger.warning(f"valid must be boolean, got {type(parsed['valid'])}")
        return False

    if not isinstance(parsed["issues"], list):
        logger.warning(f"issues must be list, got {type(parsed['issues'])}")
        return False

    return True


def safe_parse(text: str) -> Optional[Any]:
    """
    Parse JSON response safely.

    Args:
        text: Response from LLM

    Returns:
        Parsed JSON object or None if parsing fails
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
