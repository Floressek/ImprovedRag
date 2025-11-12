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
            provider: LLM provider ("api", "ollama", "huggingface", etc.) - defaults to settings
        """
        # Auto-detect provider if not specified
        if provider is None:
            provider = settings.llm.provider

        # Use simplified prompt for local models
        use_simplified = provider in ["ollama", "huggingface", "vllm"]

        if use_simplified:
            logger.info("Using SIMPLIFIED correction prompt for local model")
            corrected = self._correct_simplified(query, original_answer, verifications, contexts)
        else:
            logger.info("Using FULL correction prompt for API model")
            corrected = self._correct_full(query, original_answer, verifications, contexts, provider)

        logger.info(f"Corrected answer: {corrected[:200]}...")
        return corrected

    def _correct_simplified(
            self,
            query: str,
            original_answer: str,
            verifications: List[Verification],
            contexts: List[Dict[str, Any]],
    ) -> str:
        """Simplified correction for local models (8B-14B)."""
        failed = [v for v in verifications if v.label in ["refutes", "insufficient"]]

        if not failed:
            return original_answer

        # Build SIMPLE prompt - only what's needed
        system = """You are a fact checker. Remove false information from the answer.
CRITICAL: Answer in THE SAME LANGUAGE as the original answer!"""

        # List ONLY the false claims (no verified ones to save tokens)
        false_claims = "\n".join([
            f"{i+1}. {v.claim.text} - {v.reasoning}"
            for i, v in enumerate(failed)
        ])

        # Include ONLY relevant contexts (top 5 by score)
        sorted_contexts = sorted(
            contexts,
            key=lambda x: x.get("rerank_score") or x.get("retrieval_score") or 0.0,
            reverse=True
        )[:5]

        contexts_str = "\n\n".join([
            f"[{i+1}] {ctx.get('text', '')[:300]}"  # Truncate to 300 chars
            for i, ctx in enumerate(sorted_contexts)
        ])

        prompt = f"""{system}

QUESTION: {query}

ORIGINAL ANSWER (contains errors):
{original_answer}

FALSE STATEMENTS (remove these):
{false_claims}

SOURCES (use to fix errors):
{contexts_str}

TASK: Rewrite the answer removing false statements. Keep everything else. Use citations [N].

CORRECTED ANSWER:"""

        try:
            corrected = self.llm.generate(
                prompt=prompt,
                temperature=0.7,  # Higher for local models
                max_new_tokens=2048,  # Much lower for local
                chain_of_thought_enabled=False,  # Disable CoT for local
            ).strip()

            # Fallback if model outputs garbage
            if len(corrected) < 50 or "I'm here to assist" in corrected or "Could you please" in corrected:
                logger.warning("Local model failed to correct - returning original answer")
                return original_answer

            return corrected

        except Exception as e:
            logger.error(f"Correction failed: {e} - returning original answer")
            return original_answer

    def _correct_full(
            self,
            query: str,
            original_answer: str,
            verifications: List[Verification],
            contexts: List[Dict[str, Any]],
            provider: str,
    ) -> str:
        """Full correction for API models (32B+)."""
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

        try:
            if provider != settings.llm.provider:
                api_llm = LLMInference(provider=provider)
                corrected = api_llm.generate(
                    prompt=prompt,
                    temperature=settings.cove.temperature + 0.3,
                    max_new_tokens=8192,  # Reduced from 16192
                    chain_of_thought_enabled=True,
                ).strip()
            else:
                corrected = self.llm.generate(
                    prompt=prompt,
                    temperature=settings.cove.temperature + 0.3,
                    max_new_tokens=8192,  # Reduced from 16192
                    chain_of_thought_enabled=True,
                ).strip()

            return corrected

        except Exception as e:
            logger.error(f"API correction failed: {e} - returning original answer")
            return original_answer
