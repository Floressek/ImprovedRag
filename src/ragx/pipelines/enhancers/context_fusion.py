from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Any

from src.ragx.pipelines.enhancers.base import Enhancer

logger = logging.getLogger(__name__)

ResultT = Tuple[str, Dict[str, Any], float]


class ContextFusionEnhancer(Enhancer):
    """Merge and deduplicate contexts from multiple sub-queries (multihop)."""

    def __init__(
            self,
            deduplication_threshold: float = 0.95,
            max_contexts_per_subquery: int = 100,
    ):
        """
        Initialize context fusion enhancer.

        Args:
            deduplication_threshold: Similarity threshold for dedup (not used yet, placeholder)
            max_contexts_per_subquery: Max results to keep per sub-query
        """

        self.deduplication_threshold = deduplication_threshold
        self.max_contexts_per_subquery = max_contexts_per_subquery
        logger.info("ContextFusionEnhancer initialized")

    def process(
            self,
            query: str,
            results_by_subquery: Dict[str, List[ResultT]],
    ) -> List[ResultT]:
        """Merge results from multiple sub-queries.

        Args:
            query: Original query (not used)
            results_by_subquery: Dict mapping sub-query -> results

        Returns:
            Merged and deduplicated results
        """
        if not results_by_subquery:
            return []

        all_results = []
        seen_ids = set()

        for subquery, results in results_by_subquery.items():
            for rank, (doc_id, payload, score) in enumerate(results[:self.max_contexts_per_subquery]):
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)

                    enriched_payload = {**payload}
                    enriched_payload["source_subquery"] = subquery
                    enriched_payload["source_rank"] = rank

                    all_results.append((doc_id, enriched_payload, score))

        all_results.sort(key=lambda x: x[2], reverse=True)

        logger.info(
            f"Fused {len(all_results)} unique results from {len(results_by_subquery)} sub-queries"
        )

        return all_results

    @property
    def name(self) -> str:
        return "context_fusion"
