from __future__ import annotations

import logging
from typing import Tuple, Dict, Any, Optional, List

from src.ragx.pipelines.enhancers.base import Enhancer
from src.ragx.retrieval.constants import QUERY_TYPE_WEIGHTS, get_query_type_weight
from src.ragx.retrieval.rerankers.reranker import Reranker
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)

ResultT = Tuple[str, Dict[str, Any], float]


class MultihopRerankerEnhancer(Enhancer):
    """
    Three-stage reranking for multihop queries:
    1. Local rerank: each sub-query gets its own reranking
    2. Fusion: merge results by doc_id with score aggregation
    3. Global rerank: light reranking against original query
    """

    def __init__(
            self,
            reranker: Optional[Reranker] = None,
            top_k_per_subquery: Optional[int] = None,
            final_top_k: Optional[int] = None,
            fusion_strategy: Optional[str] = None,
            global_rerank_weight: Optional[float] = None,
    ):
        """
        Initialize multihop reranker.

        Args:
            reranker: Reranker instance
            top_k_per_subquery: How many docs to keep per sub-query after local rerank
            final_top_k: Final top-k after global rerank
            fusion_strategy: How to aggregate scores for duplicate docs ("max", "mean", "weighted_mean")
            global_rerank_weight: Weight for global rerank score (1-w for local scores)
        """
        self.reranker = reranker or Reranker()
        self.top_k_per_subquery = top_k_per_subquery or settings.retrieval.context_top_n
        self.final_top_k = final_top_k or settings.retrieval.context_top_n
        self.fusion_strategy = fusion_strategy or settings.multihop.fusion_strategy
        self.global_rerank_weight = global_rerank_weight or settings.multihop.global_rerank_weight

        logger.info(
            f"MultihopRerankerEnhancer initialized "
            f"(per_subquery={self.top_k_per_subquery}, "
            f"final={self.final_top_k}, "
            f"fusion={self.fusion_strategy}, "
            f"global_weight={self.global_rerank_weight})"
        )

    def process(
            self,
            original_query: str,
            results_by_subquery: Dict[str, List[ResultT]],
            override_top_k: Optional[int] = None,
            query_type: Optional[str] = None,
    ) -> List[ResultT]:
        """
         Process multihop results with three-stage reranking.

         Args:
             original_query: Original complex query
             results_by_subquery: Dict mapping sub-query -> retrieval results
             override_top_k: Override final top-k
             query_type: Query type example: "simple" or "complex" etc

         Returns:
             Final reranked and merged results
         """
        if not results_by_subquery:
            return []

        # Adaptive top_k: num_subqueries * 2 (minimum)
        num_subqueries = len(results_by_subquery)
        if settings.multihop.adaptive_top_k and override_top_k is None:
            adaptive_k = max(self.final_top_k, num_subqueries * 2)
            logger.info(f"Adaptive top_k: {adaptive_k} (num_subqueries={num_subqueries})")
            effective_top_k = adaptive_k
        else:
            effective_top_k = override_top_k if override_top_k is not None else self.final_top_k

        # Stage 1: Local reranking per subquery
        local_reranked = self._local_rerank(results_by_subquery, override_top_k)

        # Stage 2: Fusion with score agg.
        fused = self._fuse_results(local_reranked)

        logger.info(
            f"After fusion: {len(fused)} unique docs from "
            f"{sum(len(v) for v in local_reranked.values())} total"
        )

        # Stage 3: Global reranking with diversity (if enabled)
        if settings.multihop.diversity_enabled:
            final = self._global_rerank_with_diversity(
                original_query=original_query,
                fused_results=fused,
                local_reranked=local_reranked,
                effective_top_k=effective_top_k,
                query_type=query_type
            )
        else:
            # Fallback to standard global rerank
            final = self._global_rerank(
                original_query=original_query,
                fused_results=fused,
                override_top_k=effective_top_k,
                query_type=query_type
            )

        return final

    def _local_rerank(
            self,
            results_by_subquery: Dict[str, List[ResultT]],
            override_top_k: Optional[int] = None,
    ) -> Dict[str, List[ResultT]]:
        """Stage 1: Rerank each sub-query independently."""
        reranked_by_subquery = {}  # maybe defaultdict?

        for sub_query, results in results_by_subquery.items():
            if not results:
                reranked_by_subquery[sub_query] = []
                continue

            documents = []
            for doc_id, payload, score in results:
                documents.append({
                    "id": doc_id,
                    "text": payload.get("text", ""),
                    "payload": payload,
                    "retrieval_score": float(score),
                })

            reranked = self.reranker.rerank(
                query=sub_query,
                documents=documents,
                top_k=override_top_k if override_top_k is not None else self.top_k_per_subquery,
                text_field="text",
            )

            output = []
            for doc, rerank_score in reranked:
                p = doc["payload"]
                p["local_rerank_score"] = float(rerank_score)
                p["retrieval_score"] = float(doc.get("retrieval_score", 0.0))
                p["source_subquery"] = sub_query
                output.append((str(doc["id"]), p, float(rerank_score)))

            reranked_by_subquery[sub_query] = output

            logger.debug(
                f"Local rerank for '{sub_query[:50]}...': "
                f"{len(results)} → {len(output)} docs"
            )

        return reranked_by_subquery

    def _fuse_results(
            self,
            reranked_by_subquery: Dict[str, List[ResultT]],
    ) -> List[ResultT]:
        """Stage 2: Merge results by doc_id with score aggregation."""
        # by doc_id
        doc_map: Dict[str, List[Tuple[Dict[str, Any], float, str]]] = {}

        for sub_query, results in reranked_by_subquery.items():
            for doc_id, payload, score in results:
                if doc_id not in doc_map:
                    doc_map[doc_id] = []
                doc_map[doc_id].append((payload, score, sub_query))

        fused_results = []
        for doc_id, entries in doc_map.items():
            payloads, scores, sub_queries = zip(*entries)

            # First payload as base (they should be identical)
            base_payload = dict(payloads[0])

            if self.fusion_strategy == "max":
                fused_score = max(scores)
            elif self.fusion_strategy == "mean":
                fused_score = sum(scores) / len(scores)
            elif self.fusion_strategy == "weighted_mean":
                weights = [1.0 / (i + 1) for i in range(len(scores))]
                total_weight = sum(weights)
                fused_score = sum(score * weight for score, weight in zip(scores, weights)) / total_weight
            else:
                fused_score = max(scores)  # Default to max

            base_payload["fusion_metadata"] = {
                "source_subqueries": list(sub_queries),
                "local_scores": list(scores),
                "num_occurrences": len(entries),
                "aggregation": self.fusion_strategy,
            }
            base_payload["fused_score"] = fused_score

            fused_results.append((doc_id, base_payload, float(fused_score)))

        fused_results.sort(key=lambda x: x[2], reverse=True)

        return fused_results

    def _global_rerank(
            self,
            original_query: str,
            fused_results: List[ResultT],
            override_top_k: Optional[int] = None,
            query_type: Optional[str] = None,
    ) -> List[ResultT]:
        """Stage 3: Global rerank against original query."""
        if not fused_results:
            return []

        # based on query type
        if query_type:
            effective_weight = get_query_type_weight(query_type, default_weight=self.global_rerank_weight)
            if query_type in QUERY_TYPE_WEIGHTS:
                logger.info(f"Using query-type-specific weight for '{query_type}': {effective_weight:.2f}")
            else:
                logger.warning(f"Unknown query_type '{query_type}', using default weight {effective_weight:.2f}")
        else:
            effective_weight = self.global_rerank_weight

        # Convert to documents format
        documents = []
        for doc_id, payload, fused_score in fused_results:
            documents.append({
                "id": doc_id,
                "text": payload.get("text", ""),
                "payload": payload,
                "fused_score": float(fused_score)
            })

        # Global reranking, no top_k for it
        reranked = self.reranker.rerank(
            query=original_query,
            documents=documents,
            top_k=None,
            text_field="text",
        )

        final = []
        for doc, global_score in reranked:
            p = doc["payload"]
            fused_score = doc.get("fused_score", 0.0)

            final_score = (
                    effective_weight * global_score +
                    (1 - effective_weight) * fused_score
            )

            p["global_rerank_score"] = float(global_score)
            p["final_score"] = float(final_score)
            p["query_type"] = query_type

            final.append((str(doc["id"]), p, float(final_score)))

        final.sort(key=lambda x: x[2], reverse=True)

        effective_top_k = override_top_k if override_top_k is not None else self.final_top_k

        if effective_top_k and len(final) > effective_top_k:
            final = final[:effective_top_k]

        logger.info(
            f"Global rerank complete: {len(fused_results)} → {len(final)} docs "
            f"(weight: {effective_weight:.2f} global, "
            f"{1 - effective_weight:.2f} local{f', type={query_type}' if query_type else ''})"
        )

        return final

    def _global_rerank_with_diversity(
            self,
            original_query: str,
            fused_results: List[ResultT],
            local_reranked: Dict[str, List[ResultT]],
            effective_top_k: int,
            query_type: Optional[str] = None,
    ) -> List[ResultT]:
        """
        Stage 3: Global rerank with per-sub-query diversity guarantee.

        Phase 1 MVP:
        - Enforce min_per_subquery (at least N docs per sub-query)
        - Enforce max_per_subquery (prevent monopoly)
        - Adaptive top_k based on num_subqueries
        """
        if not fused_results:
            return []

        min_per_sq = settings.multihop.min_per_subquery
        max_per_sq = settings.multihop.max_per_subquery

        # Get query-type-specific weight
        if query_type:
            effective_weight = get_query_type_weight(query_type, default_weight=self.global_rerank_weight)
        else:
            effective_weight = self.global_rerank_weight

        # Build doc_id → sub_queries mapping from fusion_metadata
        doc_to_subqueries = {}
        for doc_id, payload, _ in fused_results:
            subqueries = payload.get("fusion_metadata", {}).get("source_subqueries", [])
            doc_to_subqueries[doc_id] = subqueries

        # Global reranking
        documents = []
        for doc_id, payload, fused_score in fused_results:
            documents.append({
                "id": doc_id,
                "text": payload.get("text", ""),
                "payload": payload,
                "fused_score": float(fused_score)
            })

        reranked = self.reranker.rerank(
            query=original_query,
            documents=documents,
            top_k=None,
            text_field="text",
        )

        # Compute final scores (weighted)
        scored_docs = []
        for doc, global_score in reranked:
            p = doc["payload"]
            fused_score = doc.get("fused_score", 0.0)
            final_score = effective_weight * global_score + (1 - effective_weight) * fused_score

            p["global_rerank_score"] = float(global_score)
            p["final_score"] = float(final_score)
            p["query_type"] = query_type

            scored_docs.append((str(doc["id"]), p, float(final_score)))

        # Track counts per sub-query
        from collections import defaultdict
        subquery_counts = defaultdict(int)

        selected = []
        selected_ids = set()  # O(1) lookup for deduplication

        # Pass 1: Enforce minimum (prioritize sub-queries that need docs)
        for doc_id, payload, score in scored_docs:
            if len(selected) >= effective_top_k:
                break

            doc_sqs = doc_to_subqueries.get(doc_id, [])
            needs_min = any(subquery_counts[sq] < min_per_sq for sq in doc_sqs)

            if needs_min:
                selected.append((doc_id, payload, score))
                selected_ids.add(doc_id)
                for sq in doc_sqs:
                    subquery_counts[sq] += 1

        # Pass 2: Fill remaining slots (respect max_per_subquery)
        for doc_id, payload, score in scored_docs:
            if len(selected) >= effective_top_k:
                break

            # Skip if already selected (O(1) lookup)
            if doc_id in selected_ids:
                continue

            doc_sqs = doc_to_subqueries.get(doc_id, [])
            violates_max = any(subquery_counts[sq] >= max_per_sq for sq in doc_sqs)

            if not violates_max:
                selected.append((doc_id, payload, score))
                selected_ids.add(doc_id)
                for sq in doc_sqs:
                    subquery_counts[sq] += 1

        # Sort by final score
        selected.sort(key=lambda x: x[2], reverse=True)

        # Log coverage per sub-query
        logger.info(
            f"Diversity rerank: {len(fused_results)} → {len(selected)} docs "
            f"(weight: {effective_weight:.2f} global, {1 - effective_weight:.2f} local)"
        )
        logger.info(f"Coverage per sub-query (min={min_per_sq}, max={max_per_sq}):")
        for sq in sorted(local_reranked.keys()):
            count = subquery_counts[sq]
            sq_short = sq[:60] + "..." if len(sq) > 60 else sq
            status = "✓" if count >= min_per_sq else "⚠"
            logger.info(f"  {status} '{sq_short}': {count} docs")

        return selected

    @property
    def name(self) -> str:
        return "multihop_reranker"
