from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, List, Iterator

from src.ragx.generation.prompts.utils.citation_remapper import remap_citations
from src.ragx.pipelines.base import BasePipeline
from src.ragx.pipelines.enhancers.cove import CoVeEnhancer
from src.ragx.pipelines.enhancers.reranker import RerankerEnhancer
from src.ragx.pipelines.enhancers.multihop_reranker import MultihopRerankerEnhancer
from src.ragx.retrieval.analyzers.linguistic_analyzer import LinguisticAnalyzer
from src.ragx.retrieval.embedder.embedder import Embedder
from src.ragx.retrieval.rewriters.adaptive_rewriter import AdaptiveQueryRewriter
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
            linguistic_analyzer: Optional[LinguisticAnalyzer] = None,
            adaptive_rewriter: Optional[AdaptiveQueryRewriter] = None,
            reranker_enhancer: Optional[RerankerEnhancer] = None,
            multihop_reranker: Optional[MultihopRerankerEnhancer] = None,
            llm: Optional[LLMInference] = None,
            cove_enhancer: Optional[CoVeEnhancer] = None,
            initial_top_k: Optional[int] = None,
    ):
        # Embedder and vector store
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or QdrantStore(
            embedding_dim=self.embedder.get_dimension(),
            recreate_collection=False
        )

        # Adaptive compt -> query rewrite
        self.linguistic_analyzer = linguistic_analyzer or LinguisticAnalyzer()
        self.adaptive_rewriter = adaptive_rewriter or AdaptiveQueryRewriter(
            analyzer=self.linguistic_analyzer,
        )

        # retrival, reranking, context merge, multihop
        self.reranker_enhancer = reranker_enhancer or RerankerEnhancer()

        # Multihop reranking (handles fusion internally)
        self.multihop_reranker = multihop_reranker or MultihopRerankerEnhancer(
            top_k_per_subquery=settings.retrieval.context_top_n,
            final_top_k=settings.retrieval.context_top_n,
            fusion_strategy="max",
            global_rerank_weight=0.6,
        )

        # generation, llm
        self.llm = llm or LLMInference()
        self.initial_top_k = initial_top_k or settings.retrieval.top_k_retrieve
        model_name = settings.llm.model_id.lower()

        # prompt
        self.prompt_builder = PromptBuilder()
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

        self.cove_enhancer = cove_enhancer or CoVeEnhancer(
            embedder=self.embedder,
            vector_store=self.vector_store,
            reranker=self.reranker_enhancer,
        )

        logger.info(
            f"EnhancedPipeline initialized "
            f"(retrieve={self.initial_top_k}, rerank_to={self.reranker_enhancer.top_k})"
            f"cove={settings.cove.enabled})"
        )

    def answer(
            self,
            query: str,
            chat_history: Optional[List[Dict[str, str]]] = None,
            max_history: Optional[int] = None,
            top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate answer for a query."""
        max_history = max_history or settings.chat.max_history
        start = time.time()

        # Step 1: Query Rewriting
        rewrite_start = time.time()
        rewrite_result = self.adaptive_rewriter.rewrite(query)
        rewrite_time = (time.time() - rewrite_start) * 1000

        original_query = rewrite_result["original"]
        queries = rewrite_result["queries"]
        is_multihop = rewrite_result["is_multihop"]
        query_type = rewrite_result.get("query_type", "general")

        logger.info(
            f"Query analysis: multihop={is_multihop}, "
            f"queries={len(queries)}, reason={rewrite_result['reasoning']}"
        )

        # Step 2: Retrieval (parallel for multihop)
        retrival_start = time.time()

        if is_multihop and len(queries) > 1:
            # Multihop: retrieve for each sub-query
            results_by_subquery = {}
            for sub_query in queries:
                qvec = self.embedder.embed_query(sub_query)
                results = self.vector_store.search(
                    vector=qvec,
                    top_k=self.initial_top_k,
                    hnsw_ef=settings.hnsw.search_ef
                )
                results_by_subquery[sub_query] = results
                logger.debug(f"Retrieved {len(results)} for sub-query: {sub_query[:50]}...")

            retrieval_time = (time.time() - retrival_start) * 1000

            total_retrieved = sum(len(v) for v in results_by_subquery.values())
            logger.info(f"Retrieved {total_retrieved} total candidates from {len(queries)} sub-queries")

            # Step 3: Multihop reranking (local → fusion → global)
            # NOTE: Fusion is handled INSIDE multihop_reranker
            rerank_start = time.time()

            results = self.multihop_reranker.process(
                original_query=original_query,
                results_by_subquery=results_by_subquery,
                override_top_k=top_k,
                query_type=query_type
            )

            rerank_time = (time.time() - rerank_start) * 1000
            logger.info(f"Retrieved {len(results)} candidates - multihop.")
            num_retrieved_candidates = total_retrieved

        else:
            query_singular = queries[0] if queries else original_query
            qvec = self.embedder.embed_query(query_singular)
            results = self.vector_store.search(
                vector=qvec,
                top_k=self.initial_top_k,
                hnsw_ef=settings.hnsw.search_ef
            )
            retrieval_time = (time.time() - retrival_start) * 1000
            logger.info(f"Retrieved {len(results)} candidates - single query.")
            num_retrieved_candidates = len(results)

            # Step 3: Standard reranking
            rerank_start = time.time()
            final_top_k = top_k if top_k is not None else self.reranker_enhancer.top_k

            original_reranker_top_k = self.reranker_enhancer.top_k

            if top_k is not None:
                self.reranker_enhancer.top_k = final_top_k
                logger.info(f"Using custom top_k = {final_top_k}")

            results = self.reranker_enhancer.process(original_query, results)  # ?

            self.reranker_enhancer.top_k = original_reranker_top_k
            rerank_time = (time.time() - rerank_start) * 1000
            logger.info(f"Reranked {len(results)} candidates")

        # Step 4: Format Contexts, TODO function for it
        contexts = []
        for idx, payload, score in results:
            metadata = payload.get("metadata", {})
            context_dict = {
                "id": idx,
                "text": payload.get("text", ""),
                "doc_title": payload.get("doc_title", "Unknown"),
                "position": payload.get("position", 0),
                "total_chunks": payload.get("total_chunks", 1),
                "retrieval_score": payload.get("retrieval_score"),
                "url": metadata.get("url"),
            }

            if is_multihop:
                context_dict["local_rerank_score"] = payload.get("local_rerank_score")
                context_dict["fused_score"] = payload.get("fused_score")
                context_dict["global_rerank_score"] = payload.get("global_rerank_score")
                context_dict["final_score"] = payload.get("final_score")
                context_dict["fusion_metadata"] = payload.get("fusion_metadata")

                # For prompt grouping
                fusion_meta = payload.get("fusion_metadata", {})
                source_subqueries = fusion_meta.get("source_subqueries", [])
                if source_subqueries:
                    context_dict["source_subquery"] = source_subqueries[0]
            else:
                context_dict["rerank_score"] = payload.get("rerank_score")

            contexts.append(context_dict)

        # Step 5: Generation
        if is_multihop:
            prompt, citation_mapping = self.prompt_builder.build(
                query=query,
                contexts=contexts,
                template_name="enhanced",
                chat_history=chat_history,
                max_history=max_history,
                config=self.prompt_config,
                is_multihop=is_multihop,
                sub_queries=queries if is_multihop else None,
            )
        else:
            prompt = self.prompt_builder.build(
                query=query,
                contexts=contexts,
                template_name="enhanced",
                chat_history=chat_history,
                max_history=max_history,
                config=self.prompt_config,
                is_multihop=is_multihop,
                sub_queries=queries if is_multihop else None,
            )
            citation_mapping = None

        llm_start = time.time()
        answer = self.llm.generate(prompt)
        llm_time = (time.time() - llm_start) * 1000

        if is_multihop and citation_mapping:
            answer, contexts = remap_citations(answer, citation_mapping, contexts)
            logger.info("Remapped citations")

        # Step 6: CoVe verification
        cove_start = time.time()
        cove_result = None
        final_answer = answer

        if settings.cove.enabled:
            logger.info("Starting CoVe verification and correction...")
            cove_result = self.cove_enhancer.verify(
                query=query,
                answer=answer,
                contexts=contexts,
            )

            if cove_result.corrected_answer:
                final_answer = cove_result.corrected_answer
                logger.info(f"Using corrected answer (status: {cove_result.status})")

        # Step 6a: Check for evidences, TODO function for it
        all_evidences = cove_result.metadata.get("all_evidences", [])
        if all_evidences:
            logger.info(f"Found {len(all_evidences)} evidences")

            existing_ids = {ctx.get("id") for ctx in contexts
                            if ctx.get("id") is not None}

            new_evidences_added = 0
            for ev in all_evidences:
                if ev.get("id") and ev["id"] not in existing_ids:
                    contexts.append({
                        "id": ev["id"],
                        "text": ev["text"],
                        "doc_title": ev.get("doc_title", "Unknown"),
                        "url": ev.get("url", ""),
                        "retrieval_score": ev["score"],
                        "source": "cove_recovery",
                        "position": ev.get("position", 0)
                    })
                    existing_ids.add(ev["id"])
                    new_evidences_added += 1

                if new_evidences_added > 0:
                    logger.info(f"Added {new_evidences_added} evidences to the context")
                else:
                    logger.info("No new evidences added")

        cove_time = (time.time() - cove_start) * 1000
        total_time = (time.time() - start) * 1000

        metadata = {
            "pipeline": "enhanced",
            "is_multihop": is_multihop,
            "sub_queries": queries if is_multihop else None,
            "reasoning": rewrite_result["reasoning"],
            "phases": [
                "linguistic_analysis",
                "adaptive_rewriting",
                "retrieval",
                "multihop_reranking (local→fusion→global)" if is_multihop else "reranking",
                "generation",
                "cove_verification" if settings.cove.enabled else None,
            ],
            "rewrite_time_ms": round(rewrite_time, 2),
            "retrieval_time_ms": round(retrieval_time, 2),
            "rerank_time_ms": round(rerank_time, 2),
            "llm_time_ms": round(llm_time, 2),
            "cove_time_ms": round(cove_time, 2) if settings.cove.enabled else 0,
            "total_time_ms": round(total_time, 2),
            "num_candidates": num_retrieved_candidates,
            "num_sources": len(contexts),
        }

        if cove_result:
            metadata["cove"] = {
                "status": cove_result.status,
                "needs_correction": cove_result.needs_correction,
                **cove_result.metadata,
            }

        return {
            "answer": final_answer,
            "sources": contexts,
            "metadata": metadata,
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
        prompt = self.prompt_builder.build(
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
