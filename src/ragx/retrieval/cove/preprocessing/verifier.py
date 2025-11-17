from __future__ import annotations

import logging
from typing import List, Dict, Any

from src.ragx.retrieval.cove.constants.types import Claim, Evidence, Verification
from src.ragx.retrieval.cove.preprocessing.validators import (
    validate_nli_response,
    validate_batch_nli_response,
)
from src.ragx.retrieval.rewriters.tools.parse import safe_parse, JSONValidator
from src.ragx.generation.inference import LLMInference
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


class ClaimVerifier:
    """Verify claims against evidence using NLI."""

    def __init__(
            self,
            llm: LLMInference,
            prompts: Dict[str, Any],
            json_validator: JSONValidator,
    ):
        self.llm = llm
        self.prompts = prompts
        self.json_validator = json_validator

    def verify(
            self,
            claims: List[Claim],
            contexts: List[Dict[str, Any]]
    ) -> List[Verification]:
        """
        Verify claims against evidence using NLI.

        Args:
            claims: List of claims to verify
            contexts: List of contexts to use as evidence

        Returns:
            List of verification results
        """
        # logger.info(f"Original claims: {claims[:50]}")
        # logger.info(f"Contexts: {contexts[:100]}")

        max_evidence = 30
        if len(contexts) > max_evidence:
            logger.warning(
                f"Too many contexts ({len(contexts)}) for verification. "
                f"Truncating to top {max_evidence} by retrieval score."
            )
            # Sort by retrieval_score or rerank_score
            sorted_contexts = sorted(
                contexts,
                key=lambda x: x.get("rerank_score") or x.get("retrieval_score") or 0.0,
                reverse=True
            )
            contexts = sorted_contexts[:max_evidence]
        evidence_texts = [
            f"[{i + 1}] {ctx.get('doc_title', 'Unknown')}: {ctx.get('text', '')}"
            for i, ctx in enumerate(contexts)
        ]
        evidence_str = "\n\n".join(evidence_texts)

        total_chars = sum(len(ctx.get('text', '')) for ctx in contexts)
        logger.info(
            f"Evidence for verification: {len(contexts)} docs, "
            f"{total_chars} chars (~{total_chars // 4} tokens)"
        )

        if settings.cove.use_batch_nli and len(claims) > 1:
            return self._batch_verify(claims, evidence_str, contexts)

        verifications = []
        for claim in claims:
            verification = self._verify_single(claim, evidence_str, contexts)
            verifications.append(verification)

        return verifications

    def _verify_single(
            self,
            claim: Claim,
            evidence_str: str,
            contexts: List[Dict[str, Any]],
    ) -> Verification:
        """Verify a single claim."""
        prompt_config = self.prompts["nli_verdict"]
        system = prompt_config["system"]
        template = prompt_config["template"]

        prompt = f"{system}\n\n{template}".format(
            claim=claim.text,
            evidence=evidence_str,
        )

        def generate_response() -> str:
            return self.llm.generate(
                prompt=prompt,
                temperature=settings.cove.temperature,
                max_new_tokens=settings.cove.max_tokens,
                chain_of_thought_enabled=False,
            ).strip()

        success, result, metadata = self.json_validator.validate_with_retry(
            generator_func=generate_response,
            parse_func=safe_parse,
            validator_func=validate_nli_response,
        )

        if not success or not result:
            logger.warning(f"Failed to verify claim {claim.text} after {metadata['attempts']} attempts")
            return Verification(
                claim=claim,
                label="insufficient",
                confidence=0.0,
                reasoning="Verification failed.",
                evidences=[],
            )

        evidence = [
            Evidence(
                doc_id=str(ctx.get("id", f"unknown_{i}")),
                text=ctx.get("text", ""),
                score=(
                    ctx.get("retrieval_score") if ctx.get("retrieval_score") is not None
                    else ctx.get("rerank_score") if ctx.get("rerank_score") is not None
                    else ctx.get("score", 0.0)
                ),
                doc_title=ctx.get("doc_title"),
                metadata={
                    "position": ctx.get("position", 0),
                    "url": ctx.get("url", ""),
                    "total_chunks": ctx.get("total_chunks", 1),
                    **(ctx.get("metadata", {}) or {}),
                },
            )
            for i, ctx in enumerate(contexts)
        ]

        return Verification(
            claim=claim,
            label=result["label"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            evidences=evidence,
        )

    def _batch_verify(
            self,
            claims: List[Claim],
            evidence_str: str,
            contexts: List[Dict[str, Any]],
    ) -> List[Verification]:
        """Verify claims against evidence using NLI."""
        prompt_config = self.prompts["batch_nli_verdict"]
        system = prompt_config["system"]
        template = prompt_config["template"]

        # Format claims with IDs
        claims_list = "\n".join([
            f"{i}. {claim.text}"
            for i, claim in enumerate(claims)
        ])

        prompt = f"{system}\n\n{template}".format(
            num_claims=len(claims),
            claims_list=claims_list,
            evidence=evidence_str,
        )

        logger.info(f"Batch verification prompt: {prompt}")

        def generate_response() -> str:
            return self.llm.generate(
                prompt=prompt,
                temperature=settings.cove.temperature,
                max_new_tokens=settings.cove.max_tokens,
                chain_of_thought_enabled=False,
            ).strip()

        success, result, metadata = self.json_validator.validate_with_retry(
            generator_func=generate_response,
            parse_func=safe_parse,
            validator_func=validate_batch_nli_response,
        )

        if not success or not result:
            logger.warning(f"Failed to batch verify claims after {metadata['attempts']} attempts")
            return [self._verify_single(claim, evidence_str, contexts) for claim in claims]

        if len(result["results"]) != len(claims):
            logger.warning(
                f"Verification results count ({len(result['results'])}) does not match claims count ({len(claims)})")
            return [self._verify_single(claim, evidence_str, contexts) for claim in claims]

        verifications = []
        evidence_objs = [
            Evidence(
                doc_id=str(ctx.get("id", f"unknown_{i}")),
                text=ctx.get("text", ""),
                score=(
                    ctx.get("retrieval_score") if ctx.get("retrieval_score") is not None
                    else ctx.get("rerank_score") if ctx.get("rerank_score") is not None
                    else ctx.get("score", 0.0)
                ),
                doc_title=ctx.get("doc_title"),
                metadata={
                    "position": ctx.get("position", 0),
                    "url": ctx.get("url", ""),
                    "total_chunks": ctx.get("total_chunks", 1),
                    **ctx.get("metadata", {}),
                },
            )
            for i, ctx in enumerate(contexts)
        ]

        for r in result["results"]:
            claim_id = r["claim_id"]
            if claim_id >= len(claims):
                continue

            verifications.append(Verification(
                claim=claims[claim_id],
                label=r["label"],
                confidence=r["confidence"],
                reasoning=r.get("reasoning", ""),
                evidences=evidence_objs,
            ))

        return verifications
