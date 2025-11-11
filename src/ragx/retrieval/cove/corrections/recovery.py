from __future__ import annotations

import logging
from typing import List, Dict, Any

from src.ragx.retrieval.cove.constants.types import Verification
from src.ragx.retrieval.cove.preprocessing.validators import validate_targeted_queries_response
from src.ragx.retrieval.rewriters.tools.parse import safe_parse, JSONValidator
from src.ragx.generation.inference import LLMInference
from src.ragx.retrieval.embedder.embedder import Embedder
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore
from src.ragx.retrieval.rerankers.reranker import Reranker
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


class RecoveryEngine:
    """Recovery phase: generate targeted queries and retrieve additional evidence."""

    def __init__(
            self,
            llm: LLMInference,
            embedder: Embedder,
            reranker: Reranker,
            vector_store: QdrantStore,
            prompts: Dict[str, Any],
            validator: JSONValidator,
    ):
        self.llm = llm
        self.embedder = embedder
        self.reranker = reranker
        self.vector_store = vector_store
        self.prompts = prompts
        self.validator = validator

    def recover(
            self,
            original_query: str,
            failed_verifications: List[Verification],
    ) -> List[Dict[str, Any]]:
        """
        Generate targeted queries and retrieve additional evidence.

        Args:
            original_query: Original query
            failed_verifications: List of failed verifications

        Returns:
            List of retrieved evidence
        """
        failed_claims = [v.claim.text for v in failed_verifications]

        targeted_queries = self._generate_queries(
            original_query=original_query,
            failed_claims=failed_claims
        )

        if not targeted_queries:
            logger.warning("No targeted queries generated during recovery.")
            return []

        logger.info(f"Generated {len(targeted_queries)} targeted queries during recovery.")

        all_evidence = []
        for query in targeted_queries[:settings.cove.max_targeted_queries]:
            evidence = self._retrieve_for_query(query)
            all_evidence.extend(evidence)

        # evidence reranking TODO CHECK FOR POSSIBLE ISSUES WITH MULTIHOP RERANKER
        if all_evidence:
            reranked = self.reranker.rerank(
                query=original_query,
                documents=all_evidence,
                top_k=settings.retrieval.top_k_retrieve,
                text_field="text"
            )
            all_evidence = [doc for doc, score in reranked]

        return all_evidence

    def _generate_queries(
            self,
            original_query: str,
            failed_claims: List[str],
    ) -> List[str]:
        """Generate targeted queries based on failed verifications."""
        prompt_config = self.prompts["targeted_queries"]
        system = prompt_config["system"]
        template = prompt_config["template"]

        missing_claims_str = "\n".join(
            f"{i + 1}. {claim}"
            for i, claim in enumerate(failed_claims)
        )

        prompt = f"{system}\n\n{template}".format(
            original_query=original_query,
            missing_claims=missing_claims_str,
        )

        def generate_response() -> str:
            return self.llm.generate(
                prompt=prompt,
                temperature=settings.cove.temperature,
                max_new_tokens=settings.cove.max_tokens,
                chain_of_thought_enabled=False,
            ).strip()

        success, result, metadata = self.validator.validate_with_retry(
            generator_func=generate_response,
            parse_func=safe_parse,
            validator_func=validate_targeted_queries_response,
        )

        if not success or not result:
            logger.warning(f"Failed to generate targeted queries during recovery: {metadata}")
            # fallback claims as queries
            return failed_claims[:3]

        return result["queries"]

    def _retrieve_for_query(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve evidence for a given query."""
        qvec = self.embedder.embed_query(query)

        results = self.vector_store.search(
            vector=qvec,
            top_k=settings.retrieval.top_k_retrieve,
            hnsw_ef=settings.hnsw.search_ef,
        )

        evidence = []
        for doc_id, payload, score in results:
            evidence.append({
                "id": doc_id,
                "text": payload.get("text", ""),
                "doc_title": payload.get("doc_title"),
                "retrieval_score": float(score),
                "metadata": payload.get("metadata", {})
            })

        return evidence
