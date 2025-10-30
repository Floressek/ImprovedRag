from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Any, Optional

from src.ragx.pipelines.enhancers.base import Enhancer
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
            fusion_strategy: str = "max",
            global_rerank_weight: float = 0.6,
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
        self.fusion_strategy = fusion_strategy
        self.global_rerank_weight = global_rerank_weight

        logger.info(
            f"MultihopRerankerEnhancer initialized "
            f"(per_subquery={self.top_k_per_subquery}, "
            f"final={self.final_top_k}, "
            f"fusion={fusion_strategy}, "
            f"global_weight={global_rerank_weight})"
        )
