from __future__ import annotations

import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.ragx.retrieval.cove.constants.cove_status import CoVeStatus
from src.ragx.retrieval.cove.constants.types import CoVeResult
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

        # TODO citation injection here

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
        if failed_verification and settings.cove.enable_recovery:
            logger.info("Attempting recovery for failed verifications...")
            additional_evidence  = self.recovery.recover(
                original_query=query,
                failed_verifications=failed_verification,
            )

            reverified = self.verifier.verify(
                [v.claim for v in failed_verification],
                additional_evidence
            )

            # update verifications
            for i, v in enumerate(failed_verification):
                if reverified[i].label == "supports":
                    idx = verifications.index(v)
                    verifications[idx] = reverified[i]

        # Step 5: Citation injection
        if missing_citations:
            logger.info(f"Injecting citations for {len(missing_citations)} claims...")
            for v in missing_citations:
                injected = self.citation_injector.inject(v.claim, contexts)
                if injected:
                    v.claim.has_citations = True
                    v.claim.citations = injected


        # Step 6: Determine status
        status = self._determine_status(verifications)

        # Step 7: Correction (if needed)
        corrected_answer = None
        needs_correction = status in [
            CoVeStatus.MISSING_EVIDENCE,
            CoVeStatus.LOW_CONFIDENCE,
            CoVeStatus.CRITICAL_FAILURE,
        ]

        if needs_correction:
            logger.info(f"Correcting answer (status: {status})")
            corrected_answer = self.corrector.correct(
                query=query,
                original_answer=answer,
                verifications=verifications,
                contexts=contexts,
            )

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
                "citations_injected": len(missing_citations),
            },
        )

    def _determine_status(self, verifications: List) -> CoVeStatus:
        """Determine overall CoVe status."""
        if not verifications:
            return CoVeStatus.SKIPPED

        num_refuted = sum(1 for v in verifications if v.label == "refutes")
        num_insufficient = sum(1 for v in verifications if v.label == "insufficient")
        num_low_conf = sum(
            1 for v in verifications
            if v.confidence < settings.cover.verification_threshold
        )

        total = len(verifications)
        refuted_ratio = num_refuted / total if total > 0 else 0
        insufficient_ratio = num_insufficient / total if total > 0 else 0

        # Critical failure
        if refuted_ratio > settings.cover.critical_failure_threshold:
            return CoVeStatus.CRITICAL_FAILURE

        # Missing evidence
        if insufficient_ratio > settings.cover.missing_evidence_threshold:
            return CoVeStatus.MISSING_EVIDENCE

        # Low confidence
        if num_low_conf > total * 0.3:
            return CoVeStatus.LOW_CONFIDENCE

        return CoVeStatus.ALL_VERIFIED




