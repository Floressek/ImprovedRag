from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_claims_response(parsed: Any) -> bool:
    """Validate the structure of a claims response."""
    if not isinstance(parsed, dict):
        logger.warning(f"Expected dict, got {type(parsed)}")
        return False

    if "claims" not in parsed:
        logger.warning("Missing 'claims' key in response")
        return False

    if not isinstance(parsed["claims"], list):
        logger.warning(f"claims must be list, got {type(parsed['claims'])}")
        return False

    if not parsed["claims"]:
        logger.warning("Empty claims list")
        return False

    return True


def validate_nli_response(parsed: Any) -> bool:
    """Validate the structure of a NLI response."""
    if not isinstance(parsed, dict):
        logger.warning(f"Expected dict, got {type(parsed)}")
        return False

    required_fields = ["label", "confidence", "reasoning"]
    for field in required_fields:
        if field not in parsed:
            logger.warning(f"Missing '{field}' key in NLI response")
            return False

    valid_labels = ["supports", "refutes", "insufficient"]
    if parsed["label"] not in valid_labels:
        logger.warning(f"Invalid label: {parsed['label']}, expected one of {valid_labels}")
        return False

    confidence = parsed["confidence"]
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        logger.warning(f"Invalid confidence value: {confidence}")
        return False

    return True


def validate_batch_nli_response(parsed: Any) -> bool:
    """Validate the structure of a NLI response."""
    if not isinstance(parsed, dict):
        logger.warning(f"Expected dict, got {type(parsed)}")
        return False

    if "results" not in parsed:
        logger.warning("Missing 'results' key in response")
        return False

    if not isinstance(parsed["results"], list):
        logger.warning(f"results must be list, got {type(parsed['results'])}")
        return False

    # for each result
    for result in parsed["results"]:
        if not isinstance(result, dict):
            logger.warning(f"Each result must be dict, got {type(result)}")
            return False

        required_fields = ["label", "claim_id", "confidence"]
        for field in required_fields:
            if field not in result:
                logger.warning(f"Missing '{field}' key in NLI response")
                return False

        valid_labels = ["supports", "refutes", "insufficient"]
        if result["label"] not in valid_labels:
            logger.warning(f"Invalid label: {result['label']}, expected one of {valid_labels}")
            return False

    return True


def validate_targeted_queries_response(parsed: Any) -> bool:
    """Validate the structure of a targeted queries response."""
    if not isinstance(parsed, dict):
        logger.warning(f"Expected dict, got {type(parsed)}")
        return False

    if "queries" not in parsed:
        logger.warning("Missing 'queries' key in response")
        return False

    if not isinstance(parsed["queries"], list):
        logger.warning(f"queries must be list, got {type(parsed['queries'])}")
        return False

    if not parsed["queries"]:
        logger.warning("Empty queries list")
        return False

    return True
