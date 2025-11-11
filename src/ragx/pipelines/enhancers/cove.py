from __future__ import annotations

import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from src.ragx.retrieval.cove.constants.cove_status import CoVeStatus
from src.ragx.retrieval.cove.constants.types import CoVeResult, Verification
from src.ragx.retrieval.cove.corrections.citation_injector import CitationInjector
from src.ragx.retrieval.cove.corrections.corrector import AnswerCorrector
from src.ragx.retrieval.cove.corrections.recovery import RecoveryEngine
from src.ragx.retrieval.cove.preprocessing.claim_extractor import ClaimExtractor
from src.ragx.retrieval.cove.preprocessing.verifier import ClaimVerifier
from src.ragx.retrieval.rewriters.tools.parse import JSONValidator, RetryConfig
from src.ragx.generation.inference import LLMInference
from src.ragx.retrieval.embedder.embedder import Embedder
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore
from src.ragx.retrieval.rerankers.reranker import Reranker
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


class CoVeEnhancer:
    """Chain-of-Verification enhancer for reducing hallucinations."""

    def __init__(
            self,
            llm: Optional[LLMInference] = None,
            embedder: Optional[Embedder] = None,
            vector_store: Optional[QdrantStore] = None,
            reranker: Optional[Reranker] = None,
    ):
        self.llm = llm or LLMInference(
            temperature=settings.cove.temperature,
            max_new_tokens=settings.cove.max_tokens,
        )
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or QdrantStore(
            embedding_dim=self.embedder.get_dimension(),
            recreate_collection=False
        )
        self.reranker = reranker or Reranker()

        retry_config = RetryConfig(
            max_retries=3,
            initial_delay=0.5,
            max_delay=4.0,
        )
        json_validator = JSONValidator(retry_config)

        base = Path(__file__).resolve().parents[2]
        prompts_path = base / "retrieval" / "cove" / "prompts" / "cove_prompts.yaml"
        prompts = self._load_prompts(prompts_path)

        self.claim_extractor = ClaimExtractor(self.llm, prompts, json_validator)
        self.verifier = ClaimVerifier(self.llm, prompts, json_validator)
        self.recovery = RecoveryEngine(
            self.llm, self.embedder, self.reranker,
            self.vector_store, prompts, json_validator
        )
        self.citation_injector = CitationInjector(self.reranker)
        self.corrector = AnswerCorrector(self.llm, prompts)

        logger.info(f"CoVeEnhancer initialized (enabled={settings.cove.enabled})")

    def _load_prompts(self, path: Path) -> Dict[str, Any]:
        """Load CoVe prompts from YAML."""
        if not path.exists():
            raise FileNotFoundError(f"Prompts file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)

        logger.info(f"Loaded CoVe prompts from {path}")
        return prompts

    def verify(
            self,
            query: str,
            answer: str,
            contexts: List[Dict[str, Any]],
    ) -> CoVeResult:
        """
        Main CoVe verification pipeline.

        Args:
            query: Original query
            answer: Generated answer to verify
            contexts: Retrieved contexts (sources)

        Returns:
            CoVeResult with verification results
        """
        if not settings.cove.enabled:
            return CoVeResult(
                original_answer=answer,
                corrected_answer=None,
                verifications=[],
                status=CoVeStatus.SKIPPED,
                needs_correction=False,
                metadata={"cove_enabled": False},
            )
        logger.info("Starting CoVe verification...")

        # Step 1: Extract claims
        claims = self.claim_extractor.extract(answer)
        logger.info(f"Extracted {len(claims)} claims from answer")

        enriched_answer = answer
        enrichment_applied = False

        if not claims:
            logger.warning("No claims extracted - checking if citation enrichment needed")
            has_citations = bool(re.search(r'\[\d+\]', answer))

            if not has_citations:
                logger.info("No citations found - applying sentence-level citation injection")
                enriched_answer, enrichment_applied = self.citation_injector.enrich_with_citations(
                    answer, contexts
                )

                if enrichment_applied:
                    logger.info("Citations injected into answer - re-extracting claims")
                    claims = self.claim_extractor.extract(enriched_answer)
                    logger.info(f"Extracted {len(claims)} claims from enriched answer")

        if not claims:
            logger.warning("No claims extracted after enrichment - returning original answer")
            return CoVeResult(
                original_answer=answer,
                corrected_answer=None,
                verifications=[],
                status=CoVeStatus.NO_CLAIMS,
                needs_correction=False,
                metadata={
                    "cove_enabled": True,
                    "enrichment_applied": enrichment_applied,
                }
            )

        # Step 2: Verify claims
        verifications = self.verifier.verify(claims, contexts)

        # Step 3: Analyze results
        failed_verification = [
            v for v in verifications
            if v.label in ["refutes", "insufficient"]
        ]

        missing_citations = [
            v for v in verifications
            if v.label == "supports" and not v.claim.has_citations
        ]

        # Step 4: Recovery (if enabled)
        recovery_attempted = False
        recovery_helped = False

        if failed_verification and settings.cove.enable_recovery:
            recovery_attempted = True
            logger.info(f"Attempting recovery for {len(failed_verification)} failed verifications...")

            additional_evidence = self.recovery.recover(
                original_query=query,
                failed_verifications=failed_verification,
            )

            if additional_evidence:
                logger.info(f"Recovery found {len(additional_evidence)} new evidence items")

                reverified = self.verifier.verify(
                    [v.claim for v in failed_verification],
                    additional_evidence
                )

                # Update verifications
                for i, v in enumerate(failed_verification):
                    if reverified[i].label == "supports":
                        idx = verifications.index(v)
                        verifications[idx] = reverified[i]
                        recovery_helped = True
                        logger.info(f"✓ Recovery SUCCESS: '{v.claim.text[:60]}...'")
                    else:
                        logger.warning(
                            f"✗ Recovery FAILED: '{v.claim.text[:60]}...' "
                            f"still {reverified[i].label} (conf: {reverified[i].confidence:.2f})"
                        )
            else:
                logger.warning("Recovery found NO evidence - marking failed claims as refuted")
                for v in failed_verification:
                    if v.label == "insufficient":
                        # Optionally change to "refutes" if no evidence found after recovery
                        idx = verifications.index(v)
                        verifications[idx] = Verification(
                            claim=v.claim,
                            label="refutes",
                            confidence=0.85,
                            reasoning="No supporting evidence found after targeted recovery attempts.",
                            evidences=v.evidences,
                        )
                        logger.info(f"Marked as REFUTES: '{v.claim.text[:60]}...'")

        # Re-count after recovery
        failed_after_recovery = [
            v for v in verifications
            if v.label in ["refutes", "insufficient"]
        ]

        missing_citations = [
            v for v in verifications
            if v.label == "supports" and not v.claim.has_citations
        ]

        logger.info(
            f"After recovery: {len(failed_after_recovery)} still failed, "
            f"{len(missing_citations)} verified but missing citations"
        )

        # Step 5-6: Determine status
        status = self._determine_status(verifications)

        # IMPORTANT: If recovery tried but failed, FORCE correction
        if recovery_attempted and not recovery_helped and failed_after_recovery:
            logger.warning(
                f"Recovery attempted but couldn't fix {len(failed_after_recovery)} claims. "
                f"Overriding status from {status} → MISSING_EVIDENCE"
            )
            status = CoVeStatus.MISSING_EVIDENCE

        # Only set MISSING_CITATIONS if NO OTHER ISSUES
        if missing_citations and status == CoVeStatus.ALL_VERIFIED:
            logger.info(f"All claims verified, but {len(missing_citations)} missing citations (metadata only)")
            status = CoVeStatus.MISSING_CITATIONS

        # Step 7: Correction (if needed)
        corrected_answer = None
        needs_correction = status.is_correction_needed() if hasattr(status, 'is_correction_needed') else status in [
            CoVeStatus.MISSING_EVIDENCE,
            CoVeStatus.LOW_CONFIDENCE,
            CoVeStatus.CRITICAL_FAILURE,
        ]

        # Works wonderfully on API, and is shitty af on local idk why
        if needs_correction:
            logger.info(f"Correcting answer (status: {status})")
            corrected_answer = self.corrector.correct(
                query=query,
                original_answer=enriched_answer if enrichment_applied else answer,
                verifications=verifications,
                contexts=contexts,
                provider=None
            )

        elif status == CoVeStatus.MISSING_CITATIONS:
            logger.info("No correction needed - only citation formatting")
            corrected_answer = enriched_answer if enrichment_applied else answer

        return CoVeResult(
            original_answer=answer,
            corrected_answer=corrected_answer,
            verifications=verifications,
            status=status,
            needs_correction=needs_correction,
            metadata={
                "num_claims": len(claims),
                "num_verified": len([v for v in verifications if v.label == "supports"]),
                "num_refuted": len([v for v in verifications if v.label == "refutes"]),
                "num_insufficient": len([v for v in verifications if v.label == "insufficient"]),
                "missing_citations": len(missing_citations),
                "enrichment_applied": enrichment_applied,
                "recovery_attempted": recovery_attempted,
                "recovery_helped": recovery_helped,
                "failed_after_recovery": len(failed_after_recovery),
            },
        )

    def _determine_status(self, verifications: List) -> CoVeStatus:
        """Determine overall CoVe status (ignoring citation formatting issues)."""
        if not verifications:
            return CoVeStatus.SKIPPED

        num_refuted = sum(1 for v in verifications if v.label == "refutes")
        num_insufficient = sum(1 for v in verifications if v.label == "insufficient")
        num_low_conf = sum(
            1 for v in verifications
            if v.confidence < settings.cove.verification_threshold
        )

        total = len(verifications)
        refuted_ratio = num_refuted / total if total > 0 else 0
        insufficient_ratio = num_insufficient / total if total > 0 else 0

        logger.debug(
            f"Status check: {total} claims → "
            f"{num_refuted} refuted ({refuted_ratio:.1%}), "
            f"{num_insufficient} insufficient ({insufficient_ratio:.1%}), "
            f"{num_low_conf} low confidence"
        )

        # Priority 1: Critical failure (too many refuted)
        if refuted_ratio > settings.cove.critical_failure_threshold:
            logger.warning(f"CRITICAL: {refuted_ratio:.1%} claims refuted (threshold: {settings.cove.critical_failure_threshold:.1%})")
            return CoVeStatus.CRITICAL_FAILURE

        # Priority 2: Missing evidence (too many insufficient)
        if insufficient_ratio > settings.cove.missing_evidence_threshold:
            logger.warning(f"MISSING EVIDENCE: {insufficient_ratio:.1%} claims insufficient (threshold: {settings.cove.missing_evidence_threshold:.1%})")
            return CoVeStatus.MISSING_EVIDENCE

        # Priority 3: ANY failed claims = at least low confidence
        if num_refuted > 0 or num_insufficient > 0:
            logger.warning(f"LOW CONFIDENCE: {num_refuted + num_insufficient} claims not verified")
            return CoVeStatus.LOW_CONFIDENCE

        # Priority 4: Low confidence scores (even if verified)
        if num_low_conf > total * 0.3:
            logger.warning(f"LOW CONFIDENCE: {num_low_conf}/{total} claims have low confidence scores")
            return CoVeStatus.LOW_CONFIDENCE

        return CoVeStatus.ALL_VERIFIED
