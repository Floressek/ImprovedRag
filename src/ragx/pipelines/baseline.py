from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, List, Iterator

from src.ragx.pipelines.base import BasePipeline
from src.ragx.retrieval.embedder.embedder import Embedder
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore
from src.ragx.generation.inference import LLMInference
from src.ragx.generation.prompts.builder import build_rag_prompt
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


class BaselinePipeline(BasePipeline):
    """Baseline RAG pipeline: Retrieval + LLM generation. -> Benchmark for the enhanced pipeline."""

    def __init__(
            self,
            embedder: Optional[Embedder] = None,
            vector_store: Optional[QdrantStore] = None,
            llm: Optional[LLMInference] = None,
            top_k: Optional[int] = None,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or QdrantStore(
            embedding_dim=self.embedder.get_dimension(),
            recreate_collection=False
        )
        self.llm = llm or LLMInference()
        self.top_k = top_k or settings.retrieval.context_top_n

        logger.info(f"BaselinePipeline initialized with (top_k = {self.top_k}).")

    def answer(
            self,
            query: str,
            chat_history: Optional[List[Dict[str, str]]] = None,
            top_k: Optional[int] = None,
            max_history: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate answer for a query."""
        top_k = top_k or self.top_k
        max_history = max_history if max_history is not None else settings.chat.max_history
        start_time = time.time()

        # Step 1: Embed query
        query_vector = self.embedder.embed_query(query)

        # Step 2: Retrieve contexts (Search)
        retrival_start = time.time()
        results = self.vector_store.search(
            vector=query_vector,
            top_k=top_k,
            hnsw_ef=settings.hnsw.search_ef
        )
        retrieval_time = (time.time() - retrival_start) * 1000  # in ms

        # Step 3: Format contexts
        contexts = []
        for idx, (doc_id, payload, score) in enumerate(results):
            metadata = payload.get("metadata", {})
            contexts.append({
                "id": doc_id,
                "text": payload.get("text", ""),
                "doc_title": payload.get("doc_title", "Unknown"),
                "url": metadata.get("url"),
                "retrieval_score": float(score),
                "rerank_score": None,
                "position": payload.get("position", 0),
                "total_chunks": payload.get("total_chunks", 1),
            })

        # Step 4: Build prompt
        prompt = build_rag_prompt(
            query=query,
            contexts=contexts,
            chat_history=chat_history,
            max_history=max_history,
        )

        # 5. Generate answer
        llm_start = time.time()
        answer = self.llm.generate(prompt)
        llm_time = (time.time() - llm_start) * 1000

        total_time = (time.time() - start_time) * 1000

        return {
            "answer": answer,
            "sources": contexts,
            "metadata": {
                "pipeline": "baseline",
                "retrieval_time_ms": round(retrieval_time, 2),
                "llm_time_ms": round(llm_time, 2),
                "total_time_ms": round(total_time, 2),
                "num_sources": len(contexts),
            }
        }
