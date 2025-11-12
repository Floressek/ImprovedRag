from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from tenacity import retry_unless_exception_type

from src.ragx.retrieval.cove.constants.types import Verification
from src.ragx.generation.inference import LLMInference
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


class AnswerCorrector:
    """Correct answers using verification results (minimal patch)."""

    def __init__(
            self,
            llm: LLMInference,
            prompts: Dict[str, Any]
    ):
        self.llm = llm
        self.prompts = prompts

    def correct(
            self,
            query: str,
            original_answer: str,
            verifications: List[Verification],
            contexts: List[Dict[str, Any]],
            provider: Optional[str] = None,
    ) -> str:
        """
        Generate corrected answer.

        Args:
            query: Original query
            original_answer: Original answer
            verification: List of verification results
            contexts: List of contexts
        """
        prompt_config = self.prompts["correct_answer"]
        system = prompt_config["system"]
        template = prompt_config["template"]

        # Format failed verifications
        failed = [v for v in verifications if v.label in ["refutes", "insufficient"]]
        failed_str = "\n\n".join([
            f"Claim {i + 1}: {v.claim.text}\n"
            f"Issue: {v.label} (confidence: {v.confidence:.2f})\n"
            f"Reasoning: {v.reasoning}"
            for i, v in enumerate(failed)
        ])

        verified = [v for v in verifications if v.label == "supports"]
        verified_str = "\n\n".join([
            f"Verified Claim {i + 1}: {v.claim.text}\n"
            f"(confidence: {v.confidence:.2f})"
            for i, v in enumerate(verified)
        ])

        contexts_str = "\n\n".join([
            f"[{i + 1}] {ctx.get('doc_title', 'Unknown')}: {ctx.get('text', '')}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = f"{system}\n\n{template}".format(
            query=query,
            answer=original_answer,
            failed_verifications_with_evidence=failed_str,
            verified_claims=verified_str,
            additional_evidence="",
            original_contexts=contexts_str
        )

        logger.info(f"Correcting answer: {prompt}")

        if provider:
            api_llm = LLMInference(provider=provider)
            corrected = api_llm.generate(
                prompt=prompt,
                temperature=settings.cove.temperature + 0.3,
                max_new_tokens=16192,
                chain_of_thought_enabled=True,
            ).strip()
        else:
            corrected = self.llm.generate(
                prompt=prompt,
                temperature=settings.cove.temperature + 0.3,
                max_new_tokens=16192,
                chain_of_thought_enabled=True,
            ).strip()



        logger.info(f"Corrected answer: {corrected}")
        return corrected
