from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, List, Iterator

from src.ragx.pipelines.base import BasePipeline
from src.ragx.pipelines.enhancers.reranker import RerankerEnhancer
from src.ragx.retrieval.embedder.embedder import Embedder
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore
from src.ragx.generation.inference import LLMInference
from src.ragx.generation.prompts.builder import PromptBuilder, PromptConfig
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


class EnhancedPipeline(BasePipeline):
    """Enhanced RAG pipeline: Retrieval → Reranking → LLM."""

    def __init__(
            self,
            embedder: Optional[Embedder] = None,
            vector_store: Optional[QdrantStore] = None,
            reranker_enhancer: Optional[RerankerEnhancer] = None,
            llm: Optional[LLMInference] = None,
            initial_top_k: Optional[int] = None,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or QdrantStore(
            embedding_dim=self.embedder.get_dimension(),
            recreate_collection=False
        )
        self.reranker_enhancer = reranker_enhancer or RerankerEnhancer()
        self.llm = llm or LLMInference()

        self.initial_top_k = initial_top_k or settings.retrieval.top_k_retrieve

        self.prompt_builder = PromptBuilder()

        model_name = settings.llm.model_id.lower()
        think_style = "qwen" if "qwen" in model_name else "none"

        self.prompt_config = PromptConfig(
            use_cot=True,
            include_metadata=True,
            strict_citations=True,
            detect_language=True,
            check_contradictions=True,
            confidence_scoring=True,
            think_tag_style=think_style,
        )

        logger.info(
            f"EnhancedPipeline initialized "
            f"(retrieve={self.initial_top_k}, rerank_to={self.reranker_enhancer.top_k})"
        )

    def answer(
            self,
            query: str,
            chat_history: Optional[List[Dict[str, str]]] = None,
            max_history: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate answer for a query."""
        max_history = max_history or settings.chat.max_history
        start = time.time()

        # Step 1: Retrieval
        query_vector = self.embedder.embed_query(query)

        retrival_start = time.time()
        results = self.vector_store.search(
            vector=query_vector,
            top_k=self.initial_top_k,
            hnsw_ef=settings.hnsw.search_ef
        )
        retrieval_time = (time.time() - retrival_start) * 1000
        logger.info(f"Retrieved {len(results)} candidates")

        # Step 2: Reranking
        rerank_start = time.time()
        results = self.reranker_enhancer.process(query, results)
        rerank_time = (time.time() - rerank_start) * 1000
        logger.info(f"Reranked {len(results)} candidates")

        # Step 3: Format Contexts
        contexts = []
        for idx, payload, score in results:
            metadata = payload.get("metadata", {})
            contexts.append({
                "id": idx,
                "text": payload.get("text", ""),
                "doc_title": payload.get("doc_title", "Unknown"),
                "position": payload.get("position", 0),
                "retrieval_score": payload.get("retrieval_score"),
                "url": metadata.get("url"),
                "rerank_score": payload.get("rerank_score"),
            })

        # Step 4: Generation
        prompt = self.prompt_builder.build(
            query=query,
            contexts=contexts,
            template_name="enhanced",
            chat_history=chat_history,
            max_history=max_history,
            config=self.prompt_config,
        )

        llm_start = time.time()
        answer = self.llm.generate(prompt)
        llm_time = (time.time() - llm_start) * 1000

        total_time = (time.time() - start) * 1000

        return {
            "answer": answer,
            "sources": contexts,
            "metadata": {
                "pipeline": "enhanced",
                "phases": ["retrieval", "reranking", "generation"],
                "retrieval_time_ms": round(retrieval_time, 2),
                "rerank_time_ms": round(rerank_time, 2),
                "llm_time_ms": round(llm_time, 2),
                "total_time_ms": round(total_time, 2),
                "num_candidates": self.initial_top_k,
                "num_sources": len(contexts),
            },
        }

    def answer_stream(
            self,
            query: str,
            chat_history: Optional[List[Dict[str, str]]] = None,
            max_history: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Generate answer for a query."""
        max_history = max_history or settings.chat.max_history
        start = time.time()

        # Step 1: Retrival
        query_vector = self.embedder.embed_query(query)

        retrival_start = time.time()
        results = self.vector_store.search(
            vector=query_vector,
            top_k=self.initial_top_k,
            hnsw_ef=settings.hnsw.search_ef
        )
        retrieval_time = (time.time() - retrival_start) * 1000
        logger.info(f"Retrieved {len(results)} candidates")

        # Step 2: Reranking
        rerank_start = time.time()
        results = self.reranker_enhancer.process(query, results)
        rerank_time = (time.time() - rerank_start) * 1000
        logger.info(f"Reranked {len(results)} candidates")

        # Step 3: Format Contexts
        contexts = []
        for idx, payload, score in results:
            contexts.append({
                "id": idx,
                "text": payload.get("text", ""),
                "doc_title": payload.get("doc_title", "Unknown"),
                "score": float(score),
                "position": payload.get("position", 0),
                "retrieval_score": payload.get("retrieval_score"),
                "rerank_score": payload.get("rerank_score"),
            })

        # Step 4: Generation
        prompt = build_rag_prompt(
            query=query,
            contexts=contexts,
            chat_history=chat_history,
            max_history=max_history,
        )

        # Yield initial metadata
        yield {
            "type": "metadata",
            "data": {
                "pipeline": "enhanced",
                "phases": ["retrieval", "reranking", "generation"],
                "retrieval_time_ms": round(retrieval_time, 2),
                "rerank_time_ms": round(rerank_time, 2),
                "num_candidates": self.initial_top_k,
                "num_sources": len(contexts),
            },
        }

        llm_start = time.time()
        for token in self.llm.generate_stream(prompt):
            yield {
                "type": "token",
                "content": token,
            }

        llm_time = (time.time() - llm_start) * 1000
        total_time = (time.time() - start) * 1000

        yield {
            "type": "done",
            "sources": contexts,
            "metadata": {
                "llm_time_ms": round(llm_time, 2),
                "total_time_ms": round(total_time, 2),
            },
        }
