from __future__ import annotations

import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, Depends

from src.ragx.api.schemas.eval import PipelineAblationRequest, PipelineAblationResponse
from src.ragx.api.dependencies import (
    get_embedder,
    get_vector_store,
    get_adaptive_rewriter,
    get_reranker_enhancer,
    get_multihop_reranker,
    get_cove_enhancer,
)
from src.ragx.generation.inference import LLMInference
from src.ragx.generation.prompts.builder import PromptBuilder, PromptConfig
from src.ragx.generation.prompts.utils.citation_remapper import remap_citations
from src.ragx.retrieval.embedder.embedder import Embedder
from src.ragx.retrieval.rewriters.adaptive_rewriter import AdaptiveQueryRewriter
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore
from src.ragx.pipelines.enhancers.reranker import RerankerEnhancer
from src.ragx.pipelines.enhancers.multihop_reranker import MultihopRerankerEnhancer
from src.ragx.pipelines.enhancers.cove import CoVeEnhancer
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["Evaluation"])


@router.post("/ablation", response_model=PipelineAblationResponse)
async def pipeline_ablation(
        request: PipelineAblationRequest,
        embedder: Embedder = Depends(get_embedder),
        vector_store: QdrantStore = Depends(get_vector_store),
        adaptive_rewriter: AdaptiveQueryRewriter = Depends(get_adaptive_rewriter),
        reranker_enhancer: RerankerEnhancer = Depends(get_reranker_enhancer),
        multihop_reranker: MultihopRerankerEnhancer = Depends(get_multihop_reranker),
        cove_enhancer: CoVeEnhancer = Depends(get_cove_enhancer),
) -> PipelineAblationResponse:
    """
    Pipeline ablation endpoint - follows enhanced.py flow with toggles.

    5 Independent Toggles:
    1. query_analysis_enabled: Enable/disable adaptive_rewriter (multihop detection)
    2. enhanced_features_enabled: Enable/disable metadata, contradictions, etc. in prompt
    3. cot_enabled: Enable/disable Chain-of-Thought
    4. reranker_enabled: Enable/disable reranking
    5. cove_mode: "off", "auto", "metadata", "suggest"
    """
    start_total = time.time()
    query = request.query

    logger.info(
        f"Ablation: query_analysis={request.query_analysis_enabled}, "
        f"enhanced={request.enhanced_features_enabled}, cot={request.cot_enabled}, "
        f"rerank={request.reranker_enabled}, cove={request.cove_mode}"
    )

    metadata = {
        "ablation_config": {
            "query_analysis_enabled": request.query_analysis_enabled,
            "enhanced_features_enabled": request.enhanced_features_enabled,
            "cot_enabled": request.cot_enabled,
            "reranker_enabled": request.reranker_enabled,
            "cove_mode": request.cove_mode,
        },
        "timings": {},
    }

    # STEP 1: Query Rewriting (Toggle 1: query_analysis_enabled)
    rewrite_start = time.time()

    if request.query_analysis_enabled:
        rewrite_result = adaptive_rewriter.rewrite(query)
        original_query = rewrite_result["original"]
        queries = rewrite_result["queries"]
        is_multihop = rewrite_result["is_multihop"]
        query_type = rewrite_result.get("query_type", "general")
        reasoning = rewrite_result.get("reasoning", "")

        logger.info(
            f"Query analysis: multihop={is_multihop}, "
            f"queries={len(queries)}, reason={reasoning}"
        )
    else:
        # Query analysis OFF - treat as single query
        original_query = query
        queries = [query]
        is_multihop = False
        query_type = "general"
        reasoning = "Query analysis disabled"

    rewrite_time = (time.time() - rewrite_start) * 1000
    metadata["timings"]["rewrite_ms"] = round(rewrite_time, 2)
    metadata["is_multihop"] = is_multihop
    metadata["sub_queries"] = queries
    metadata["query_type"] = query_type

    # STEP 2: Retrieval (parallel for multihop)
    retrieval_start = time.time()

    if is_multihop and len(queries) > 1:
        # Multihop: retrieve for each sub-query
        results_by_subquery = {}
        for sub_query in queries:
            qvec = embedder.embed_query(sub_query)
            results = vector_store.search(
                vector=qvec,
                top_k=request.top_k * 2,  # Over-retrieve for reranking
                hnsw_ef=settings.hnsw.search_ef
            )
            results_by_subquery[sub_query] = results
            logger.debug(f"Retrieved {len(results)} for sub-query: {sub_query[:50]}...")

        total_retrieved = sum(len(v) for v in results_by_subquery.values())
        logger.info(f"Retrieved {total_retrieved} total from {len(queries)} sub-queries")
        num_retrieved_candidates = total_retrieved
    else:
        # Single query retrieval
        query_singular = queries[0] if queries else original_query
        qvec = embedder.embed_query(query_singular)
        results = vector_store.search(
            vector=qvec,
            top_k=request.top_k * 2,  # Over-retrieve for reranking
            hnsw_ef=settings.hnsw.search_ef
        )
        logger.info(f"Retrieved {len(results)} candidates - single query")
        num_retrieved_candidates = len(results)
        results_by_subquery = {query_singular: results}

    retrieval_time = (time.time() - retrieval_start) * 1000
    metadata["timings"]["retrieval_ms"] = round(retrieval_time, 2)
    metadata["num_candidates"] = num_retrieved_candidates

    # Edge case: No results
    if num_retrieved_candidates == 0:
        logger.warning("No results found from vector store")
        metadata["total_time_ms"] = round((time.time() - start_total) * 1000, 2)
        return PipelineAblationResponse(
            answer="I couldn't find any relevant information to answer your question.",
            contexts=[],
            context_details=[],
            sub_queries=queries,
            sources=[],
            metadata=metadata,
        )

    # STEP 3: Reranking (Toggle 4: reranker_enabled)
    rerank_start = time.time()

    if request.reranker_enabled:
        if is_multihop and len(queries) > 1:
            # Multihop reranking (local → fusion → global)
            results = multihop_reranker.process(
                original_query=original_query,
                results_by_subquery=results_by_subquery,
                override_top_k=request.top_k,
                query_type=query_type
            )
            logger.info(f"Multihop reranked to {len(results)} candidates")
        else:
            # Standard reranking
            original_top_k = reranker_enhancer.top_k
            reranker_enhancer.top_k = request.top_k

            query_for_rerank = queries[0] if queries else original_query
            all_results = results_by_subquery.get(query_for_rerank, [])
            results = reranker_enhancer.process(original_query, all_results)

            reranker_enhancer.top_k = original_top_k
            logger.info(f"Reranked to {len(results)} candidates")
    else:
        # No reranking - take top_k from first subquery
        query_for_results = queries[0] if queries else original_query
        all_results = results_by_subquery.get(query_for_results, [])
        results = all_results[:request.top_k]
        logger.info(f"Reranking disabled - taking top {len(results)}")

    rerank_time = (time.time() - rerank_start) * 1000
    metadata["timings"]["rerank_ms"] = round(rerank_time, 2)
    metadata["num_sources"] = len(results)

    # STEP 4: Format Contexts (same as enhanced.py)
    contexts = []
    for idx, payload, score in results:
        payload_metadata = payload.get("metadata", {})
        context_dict = {
            "id": idx,
            "text": payload.get("text", ""),
            "doc_title": payload.get("doc_title", "Unknown"),
            "position": payload.get("position", 0),
            "total_chunks": payload.get("total_chunks", 1),
            "retrieval_score": payload.get("retrieval_score"),
            "url": payload_metadata.get("url"),
        }

        if is_multihop:
            context_dict["local_rerank_score"] = payload.get("local_rerank_score")
            context_dict["fused_score"] = payload.get("fused_score")
            context_dict["global_rerank_score"] = payload.get("global_rerank_score")
            context_dict["final_score"] = payload.get("final_score")
            context_dict["fusion_metadata"] = payload.get("fusion_metadata")

            fusion_meta = payload.get("fusion_metadata", {})
            source_subqueries = fusion_meta.get("source_subqueries", [])
            if source_subqueries:
                context_dict["source_subquery"] = source_subqueries[0]
        else:
            context_dict["rerank_score"] = payload.get("rerank_score")

        contexts.append(context_dict)

    # STEP 5: Generation
    llm_start = time.time()

    # Build prompt config (Toggles 2 & 3)
    prompt_config = PromptConfig(
        use_cot=request.cot_enabled,  # Toggle 3
        include_metadata=request.enhanced_features_enabled,  # Toggle 2
        strict_citations=request.enhanced_features_enabled,
        detect_language=request.enhanced_features_enabled,
        check_contradictions=request.enhanced_features_enabled,
        confidence_scoring=request.enhanced_features_enabled,
        think_tag_style="qwen" if request.cot_enabled else "none",
    )

    provider = request.provider or settings.llm.provider
    llm = LLMInference(provider=provider)
    prompt_builder = PromptBuilder()

    if is_multihop:
        prompt, citation_mapping = prompt_builder.build(
            query=query,
            contexts=contexts,
            template_name="enhanced",
            config=prompt_config,
            is_multihop=True,
            sub_queries=queries,
        )
    else:
        prompt = prompt_builder.build(
            query=query,
            contexts=contexts,
            template_name="enhanced",
            config=prompt_config,
            is_multihop=False,
            sub_queries=None,
        )
        citation_mapping = None

    answer = llm.generate(prompt, temperature=0.1, max_new_tokens=500)
    llm_time = (time.time() - llm_start) * 1000
    metadata["timings"]["llm_ms"] = round(llm_time, 2)

    # Remap citations if multihop
    if is_multihop and citation_mapping:
        answer, contexts = remap_citations(answer, citation_mapping, contexts)
        logger.info("Remapped citations for multihop")

    # STEP 6: CoVe Verification (Toggle 5: cove_mode)
    cove_start = time.time()
    final_answer = answer
    sources = contexts.copy()

    if request.cove_mode != "off":
        logger.info(f"Running CoVe in mode: {request.cove_mode}")

        # Temporarily enable CoVe
        original_cove_enabled = settings.cove.enabled
        settings.cove.enabled = True

        cove_result = cove_enhancer.verify(
            query=query,
            answer=answer,
            contexts=contexts,
            correction_mode=request.cove_mode,
        )

        settings.cove.enabled = original_cove_enabled

        if cove_result.corrected_answer:
            final_answer = cove_result.corrected_answer
            logger.info(f"Using corrected answer (status: {cove_result.status})")

        metadata["cove"] = {
            "status": str(cove_result.status),
            "needs_correction": cove_result.needs_correction,
            **cove_result.metadata,
        }

        # Merge CoVe evidences (same as enhanced.py)
        all_evidences = cove_result.metadata.get("all_evidences", [])
        if all_evidences:
            logger.info(f"Found {len(all_evidences)} evidences from CoVe")

            existing_ids = {ctx.get("id") for ctx in sources if ctx.get("id") is not None}
            new_evidences_added = 0

            for ev in all_evidences:
                ev_id = ev.get("id")
                if not ev_id or ev_id in existing_ids or str(ev_id).startswith("unknown_"):
                    continue

                sources.append({
                    "id": ev_id,
                    "text": ev["text"],
                    "doc_title": ev.get("doc_title", "Unknown"),
                    "position": ev.get("position", 0),
                    "retrieval_score": ev.get("score", 0),
                    "url": ev.get("url", ""),
                    "source": "EVIDENCE FROM COVE",
                })
                existing_ids.add(ev_id)
                new_evidences_added += 1

            if new_evidences_added > 0:
                logger.info(f"Added {new_evidences_added} new evidences to sources")

    cove_time = (time.time() - cove_start) * 1000
    metadata["timings"]["cove_ms"] = round(cove_time, 2)

    # Final metadata
    total_time = (time.time() - start_total) * 1000
    metadata["total_time_ms"] = round(total_time, 2)

    # Build response (match schema)
    contexts_text = [ctx.get("text", "") for ctx in contexts]

    context_details = []
    for ctx in contexts:
        detail = {
            "text": ctx.get("text", ""),
            "url": ctx.get("url", ""),
            "title": ctx.get("doc_title", ""),
            "score": ctx.get("rerank_score") or ctx.get("final_score") or ctx.get("retrieval_score", 0.0),
        }
        if "citation_id" in ctx:
            detail["citation_id"] = ctx["citation_id"]
        context_details.append(detail)

    sources_list = []
    seen_urls = set()
    for src in sources:
        url = src.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources_list.append({
                "url": url,
                "title": src.get("doc_title", ""),
                "citation_id": src.get("citation_id"),
            })

    return PipelineAblationResponse(
        answer=final_answer,
        contexts=contexts_text,
        context_details=context_details,
        sub_queries=queries,
        sources=sources_list,
        metadata=metadata,
    )
