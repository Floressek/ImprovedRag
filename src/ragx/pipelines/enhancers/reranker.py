from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Dict, Any

from src.ragx.pipelines.enhancers.base import Enhancer
from src.ragx.retrieval.rerankers.reranker import Reranker
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)

ResultT = Tuple[str, Dict[str, Any], float]  # (doc_id, payload, retrieval_score)


class RerankerEnhancer(Enhancer):
    """Reranking enhancer using cross-encoder."""

    def __init__(
            self,
            reranker: Optional[Reranker] = None,
            top_k: Optional[int] = None,
    ):
        self.reranker = reranker or Reranker()
        self.top_k: int = int(top_k if top_k is not None else settings.retrieval.context_top_n)
        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {self.top_k}")
        logger.info("RerankerEnhancer initialized (top_k=%s)", self.top_k)

    def process(
            self,
            query: str,
            results: List[ResultT],
            top_k: Optional[int] = None,
    ) -> List[ResultT]:
        """Rerank results using cross-encoder."""
        if not results:
            return []

        top_k = top_k or self.top_k

        k = min(top_k, len(results))
        logger.debug(f"Reranking {len(results)} results (effective top_k={k})")

        # Convert to documents format
        documents: List[Dict[str, Any]] = []
        for doc_id, payload, score in results:
            p = payload or {}
            text = p.get("text")
            if not isinstance(text, str):
                text = ""
            documents.append(
                {
                    "id": doc_id,
                    "text": text,
                    "payload": p,
                    "retrieval_score": float(score),
                }
            )

        reranked = self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=k,
            text_field="text",
        )

        # Convert back to (id, payload, score) format
        output: List[ResultT] = []
        for doc, rerank_score in reranked:
            p = doc["payload"]
            p["rerank_score"] = float(rerank_score)
            p["retrieval_score"] = float(doc.get("retrieval_score", 0.0))
            output.append((str(doc["id"]), p, float(rerank_score)))

        logger.info(f"Reranked {len(output)} results, Original articles for reranking: {len(results)}")
        return output

    @property
    def name(self) -> str:
        return "reranker"
