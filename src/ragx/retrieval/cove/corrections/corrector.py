from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from src.ragx.retrieval.cove.constants.types import Verification
from src.ragx.retrieval.cove.preprocessing.validators import validate_claims_response
from src.ragx.retrieval.rewriters.tools.parse import safe_parse, JSONValidator
from src.ragx.generation.inference import LLMInference
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


class AnswerCorrector:
    """Correct answers using verification results (minimal patch)."""

    def __init__(
            self,
            llm: LLMInference,
            prompts: Dict[str, Any],
            json_validator: JSONValidator,
    ):
        self.llm = llm
        self.prompts = prompts
        self.json_validator = json_validator

    def correct(
            self,
            query: str,
            original_answer: str,
            verifications: List[Verification],
            contexts: List[Dict[str, Any]],
            provider: Optional[str] = None,
    ) -> tuple[Optional[str], Dict[str, Any]]:
        """
        Generate corrected answer based on correction_mode.

        Args:
            query: Original query
            original_answer: Original answer
            verifications: List of verification results
            contexts: List of contexts
            provider: LLM provider ("api", "ollama", "huggingface", etc.) - defaults to settings

        Returns:
            (corrected_answer, metadata_dict)
            - For "metadata" mode: (None, {...uncertain_claims...})
            - For "suggest" mode: (corrected_or_None, {...suggestions...})
            - For "auto" mode: (corrected_answer, {...})
        """
        # Auto-detect provider if not specified
        if provider is None:
            provider = settings.llm.provider

        mode = settings.cove.correction_mode
        failed = [v for v in verifications if v.label in ["refutes", "insufficient"]]

        if not failed:
            return original_answer, {}

        logger.info(f"Correction mode: {mode}, failed claims: {len(failed)}")

        # Mode 1: metadata - tylko zwróć info o niepewnych claimach
        if mode == "metadata":
            return self._metadata_only(failed)

        # Mode 2: suggest - zaproponuj poprawki, zdecyduj czy zastosować
        elif mode == "suggest":
            return self._suggest_and_decide(query, original_answer, failed, contexts, provider)

        # Mode 3: auto - automatycznie koryguj
        else:  # auto
            return self._auto_correct(query, original_answer, failed, contexts, provider)

    def _metadata_only(
            self,
            failed_verifications: List[Verification],
    ) -> tuple[None, Dict[str, Any]]:
        """Mode: metadata - tylko info w metadata, nie koryguj."""
        uncertain_claims = [
            {
                "claim": v.claim.text,
                "issue": v.label,
                "reasoning": v.reasoning,
                "confidence": v.confidence,
            }
            for v in failed_verifications
        ]

        logger.info(f"Metadata-only mode: returning {len(uncertain_claims)} uncertain claims")
        return None, {"uncertain_claims": uncertain_claims}

    def _suggest_and_decide(
            self,
            query: str,
            original_answer: str,
            failed_verifications: List[Verification],
            contexts: List[Dict[str, Any]],
            provider: str,
    ) -> tuple[Optional[str], Dict[str, Any]]:
        """Mode: suggest - 2-step: zaproponuj → zdecyduj czy zastosować."""

        # Step 1: LLM proponuje poprawki
        suggestions = self._generate_suggestions(query, original_answer, failed_verifications, contexts)

        if not suggestions:
            logger.warning("No suggestions generated")
            return None, {"suggestions": [], "applied": False}

        # Step 2: Zdecyduj czy zastosować (na podstawie confidence)
        threshold = settings.cove.correction_confidence_threshold
        high_confidence_suggestions = [
            s for s in suggestions
            if s.get("confidence", 0.0) >= threshold
        ]

        metadata = {
            "suggestions": suggestions,
            "total_suggestions": len(suggestions),
            "high_confidence": len(high_confidence_suggestions),
            "threshold": threshold,
        }

        # Jeśli wszystkie sugestie mają wysoką pewność → zastosuj
        if len(high_confidence_suggestions) == len(suggestions):
            logger.info(f"All {len(suggestions)} suggestions have high confidence - applying corrections")
            corrected = self._apply_suggestions(original_answer, high_confidence_suggestions)
            metadata["applied"] = True
            return corrected, metadata
        else:
            logger.info(f"Only {len(high_confidence_suggestions)}/{len(suggestions)} suggestions have high confidence - not applying")
            metadata["applied"] = False
            return None, metadata

    def _generate_suggestions(
            self,
            query: str,
            original_answer: str,
            failed_verifications: List[Verification],
            contexts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate correction suggestions using LLM."""
        prompt_config = self.prompts["suggest_corrections"]
        system = prompt_config["system"]
        template = prompt_config["template"]

        false_claims = "\n".join([
            f"{i+1}. {v.claim.text} - {v.reasoning}"
            for i, v in enumerate(failed_verifications)
        ])

        # Use contexts in ORIGINAL order to preserve citation mapping
        # (contexts are already sorted by reranking in pipeline)
        # Truncate each to 500 chars for suggest mode (more detail than simplified)
        contexts_str = "\n\n".join([
            f"[{i+1}] {ctx.get('text', '')[:500]}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = f"{system}\n\n{template}".format(
            query=query,
            answer=original_answer,
            false_claims=false_claims,
            contexts=contexts_str,
        )

        def generate_response() -> str:
            return self.llm.generate(
                prompt=prompt,
                temperature=0.5,
                max_new_tokens=2048,
                chain_of_thought_enabled=False,
            ).strip()

        success, result, metadata = self.json_validator.validate_with_retry(
            generator_func=generate_response,
            parse_func=safe_parse,
            validator_func=lambda x: x.get("suggestions") is not None,
        )

        if not success or not result:
            logger.error(f"Failed to generate suggestions after {metadata['attempts']} attempts")
            return []

        return result.get("suggestions", [])

    def _apply_suggestions(
            self,
            original_answer: str,
            suggestions: List[Dict[str, Any]],
    ) -> str:
        """Apply suggestions to original answer."""
        corrected = original_answer

        for suggestion in suggestions:
            original_claim = suggestion.get("original_claim", "")
            correction = suggestion.get("suggested_correction", "")

            if correction == "REMOVE":
                # Remove the sentence containing this claim
                corrected = corrected.replace(original_claim, "")
            else:
                # Replace with correction
                corrected = corrected.replace(original_claim, correction)

        # Clean up double spaces and newlines
        corrected = " ".join(corrected.split())
        return corrected

    def _auto_correct(
            self,
            query: str,
            original_answer: str,
            failed_verifications: List[Verification],
            contexts: List[Dict[str, Any]],
            provider: str,
    ) -> tuple[str, Dict[str, Any]]:
        """Mode: auto - automatycznie koryguj."""

        # Use simplified prompt for local models
        use_simplified = provider in ["ollama", "huggingface", "vllm"]

        if use_simplified:
            logger.info("Using SIMPLIFIED correction prompt for local model")
            corrected = self._correct_simplified(query, original_answer, failed_verifications, contexts)
        else:
            logger.info("Using FULL correction prompt for API model")
            corrected = self._correct_full(query, original_answer, failed_verifications, contexts, provider)

        return corrected, {"mode": "auto", "simplified": use_simplified}

    def _correct_simplified(
            self,
            query: str,
            original_answer: str,
            failed_verifications: List[Verification],
            contexts: List[Dict[str, Any]],
    ) -> str:
        """Simplified correction for local models (8B-14B) - using YAML prompt."""
        prompt_config = self.prompts["correct_answer_simplified"]
        system = prompt_config["system"]
        template = prompt_config["template"]

        false_claims = "\n".join([
            f"{i+1}. {v.claim.text} - {v.reasoning}"
            for i, v in enumerate(failed_verifications)
        ])

        # Use contexts in ORIGINAL order to preserve citation mapping
        # (contexts are already sorted by reranking in pipeline)
        # Truncate each to 400 chars to fit in prompt
        contexts_str = "\n\n".join([
            f"[{i+1}] {ctx.get('text', '')[:400]}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = f"{system}\n\n{template}".format(
            query=query,
            answer=original_answer,
            false_claims=false_claims,
            contexts=contexts_str,
        )

        try:
            corrected = self.llm.generate(
                prompt=prompt,
                temperature=0.7,  # Higher for local models
                max_new_tokens=2048,
                chain_of_thought_enabled=False,
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
            failed_verifications: List[Verification],
            contexts: List[Dict[str, Any]],
            provider: str,
    ) -> str:
        """Full correction for API models (32B+)."""
        prompt_config = self.prompts["correct_answer"]
        system = prompt_config["system"]
        template = prompt_config["template"]

        # Format failed verifications
        failed_str = "\n\n".join([
            f"Claim {i + 1}: {v.claim.text}\n"
            f"Issue: {v.label} (confidence: {v.confidence:.2f})\n"
            f"Reasoning: {v.reasoning}"
            for i, v in enumerate(failed_verifications)
        ])

        contexts_str = "\n\n".join([
            f"[{i + 1}] {ctx.get('doc_title', 'Unknown')}: {ctx.get('text', '')}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = f"{system}\n\n{template}".format(
            query=query,
            answer=original_answer,
            failed_verifications_with_evidence=failed_str,
            verified_claims="",  # Not needed in simplified flow
            additional_evidence="",
            original_contexts=contexts_str
        )

        try:
            if provider != settings.llm.provider:
                api_llm = LLMInference(provider=provider)
                corrected = api_llm.generate(
                    prompt=prompt,
                    temperature=settings.cove.temperature + 0.3,
                    max_new_tokens=8192,
                    chain_of_thought_enabled=True,
                ).strip()
            else:
                corrected = self.llm.generate(
                    prompt=prompt,
                    temperature=settings.cove.temperature + 0.3,
                    max_new_tokens=8192,
                    chain_of_thought_enabled=True,
                ).strip()

            return corrected

        except Exception as e:
            logger.error(f"API correction failed: {e} - returning original answer")
            return original_answer
