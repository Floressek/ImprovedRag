from __future__ import annotations

import logging
import re
from typing import List, Dict, Any

from src.ragx.retrieval.cove.constants.types import Claim
from src.ragx.retrieval.cove.preprocessing.validators import validate_claims_response
from src.ragx.retrieval.rewriters.tools.parse import safe_parse, JSONValidator
from src.ragx.generation.inference import LLMInference
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)

class ClaimExtractor:
    """Extracts claims from a generated text."""

    def __init__(
            self,
            llm: LLMInference,
            prompts: Dict[str, Any],
            json_validator: JSONValidator,
    ):
        self.llm = llm
        self.prompts = prompts
        self.json_validator = json_validator

    def extract(self, answer: str) -> List[Claim]:
        """Extract claims from a generated answer.

        Args:
            answer: The generated text to extract claims from.

        Returns:
            List of extracted claims.
        """
        prompt_config = self.prompts["extract_claims"]
        system = prompt_config["system"]
        template = prompt_config["template"]

        prompt = f"{system}\n\n{template}".format(
            answer=answer,
            max_claims=settings.cove.max_verification,
        )

        def generate_response() -> str:
            return self.llm.generate(
                prompt=prompt,
                temperature=settings.cove.temperature,
                max_new_tokens=settings.cove.max_tokens,
                chain_of_thought_enabled=True,
            ).strip()

        success, result, metadata = self.json_validator.validate_with_retry(
            generator_func=generate_response,
            parse_func=safe_parse,
            validator_func=validate_claims_response,
        )

        if not success or not result:
            logger.error(f"Failed to extract claims after {metadata['attempts']} attempts")
            return []

        claims = []
        for i, claim_text in enumerate(result["claims"]):
            has_citation = bool(re.search(r'\[\d+\]', claim_text))
            citations = [int(m.group(1)) for m in re.finditer(r'\[(\d+)\]', claim_text)]

            claims.append(Claim(
                text=claim_text,
                claim_id=i,
                has_citations=has_citation,
                citations=citations,
            ))

        logger.info(f"Extracted {len(claims)} claims from answer")
        return claims
