"""Pipeline ablation endpoint for testing different configurations."""
from __future__ import annotations

import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, Depends

from src.ragx.api.schemas.eval import PipelineAblationRequest, PipelineAblationResponse
from src.ragx.api.dependencies import (
    get_embedder,
    get_vector_store,
    get_linguistic_analyzer,
    get_adaptive_rewriter,
    get_reranker_enhancer,
    get_multihop_reranker,
    get_cove_enhancer,
)
from src.ragx.generation.inference import LLMInference
from src.ragx.generation.prompts.builder import PromptBuilder, PromptConfig
from src.ragx.generation.prompts.utils.citation_remapper import remap_citations
from src.ragx.retrieval.analyzers.linguistic_analyzer import LinguisticAnalyzer
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
        linguistic_analyzer: LinguisticAnalyzer = Depends(get_linguistic_analyzer),
        adaptive_rewriter: AdaptiveQueryRewriter = Depends(get_adaptive_rewriter),
        reranker_enhancer: RerankerEnhancer = Depends(get_reranker_enhancer),
        multihop_reranker: MultihopRerankerEnhancer = Depends(get_multihop_reranker),
        cove_enhancer: CoVeEnhancer = Depends(get_cove_enhancer),
) -> Dict[str, Any]:
    """
    Pipeline ablation study - toggle each stage on/off.

    Allows testing different combinations of:
    - Query analysis & multihop decomposition
    - Reranking (standard or multihop)
    - Prompt templates (basic/enhanced/multihop/auto)
    - Chain-of-Thought
    - CoVe verification
    - LLM provider

    Args:
        request: Configuration for which stages to enable/disable

    Returns:
        Answer with detailed metadata about what was enabled/disabled
    """
    start = time.time()
    query = request.query

    logger.info(
        f"Ablation request: query_analysis={request.use_query_analysis}, "
        f"reranker={request.use_reranker}, cove={request.use_cove}, "
        f"template={request.prompt_template}, cot={request.use_cot}"
    )

    metadata = {
        "ablation_config": {
            "use_query_analysis": request.use_query_analysis,
            "use_reranker": request.use_reranker,
            "use_cove": request.use_cove,
            "use_cot": request.use_cot,
            "prompt_template": request.prompt_template,
            "provider": request.provider if request.provider else settings.llm.provider,
        }
    }

    # STEP 1: Query Analysis (optional)
    is_multihop = False
    queries = [query]
    query_type = "general"
    citation_mapping = None

    if request.use_query_analysis:
        rewrite_start = time.time()
        rewrite_result = adaptive_rewriter.rewrite(query)
        metadata["rewrite_time_ms"] = round((time.time() - rewrite_start) * 1000, 2)

        is_multihop = rewrite_result["is_multihop"]
        queries = rewrite_result["queries"]
        query_type = rewrite_result.get("query_type", "general")

        metadata["query_analysis"] = {
            "is_multihop": is_multihop,
            "num_sub_queries": len(queries),
            "query_type": query_type,
            "reasoning": rewrite_result["reasoning"],
        }
        logger.info(f"Query analysis: multihop={is_multihop}, queries={len(queries)}")
    else:
        metadata["query_analysis"] = {"skipped": True}
        logger.info("Query analysis SKIPPED")

    # STEP 2: Retrieval
    retrieval_start = time.time()

    if is_multihop and len(queries) > 1:
        # Multihop retrieval
        results_by_subquery = {}
        for sub_query in queries:
            qvec = embedder.embed_query(sub_query)
            results = vector_store.search(
                vector=qvec,
                top_k=settings.retrieval.top_k_retrieve,
                hnsw_ef=settings.hnsw.search_ef
            )
            results_by_subquery[sub_query] = results

        total_retrieved = sum(len(v) for v in results_by_subquery.values())
        metadata["retrieval"] = {
            "method": "multihop",
            "total_retrieved": total_retrieved,
            "num_sub_queries": len(queries),
        }
    else:
        # Single query retrieval
        query_singular = queries[0] if queries else query
        qvec = embedder.embed_query(query_singular)
        results = vector_store.search(
            vector=qvec,
            top_k=settings.retrieval.top_k_retrieve,
            hnsw_ef=settings.hnsw.search_ef
        )
        results_by_subquery = None
        metadata["retrieval"] = {
            "method": "single",
            "total_retrieved": len(results),
        }

    metadata["retrieval_time_ms"] = round((time.time() - retrieval_start) * 1000, 2)

    # STEP 3: Reranking (optional, but FORCED for multihop)
    rerank_start = time.time()

    if is_multihop and results_by_subquery:
        # Multihop ALWAYS needs at least local+fusion
        if request.use_reranker:
            # Full 3-stage reranking (local→fusion→global)
            final_results = multihop_reranker.process(
                original_query=query,
                results_by_subquery=results_by_subquery,
                override_top_k=request.top_k,
                query_type=query_type
            )
            metadata["reranking"] = {"method": "multihop (local→fusion→global)", "forced": False}
        else:
            # Minimal: only local+fusion (no global reranking)
            local_reranked = multihop_reranker._local_rerank(results_by_subquery)
            final_results = multihop_reranker._fuse_results(local_reranked)
            final_results = final_results[:request.top_k]
            metadata["reranking"] = {"method": "multihop (local→fusion only)", "forced": True, "note": "Multihop requires at least fusion"}
            logger.info("Multihop detected: forcing local+fusion reranking (use_reranker=false, so skipping global)")
    elif request.use_reranker:
        # Standard single-query reranking
        original_top_k = reranker_enhancer.top_k
        reranker_enhancer.top_k = request.top_k
        final_results = reranker_enhancer.process(query, results)
        reranker_enhancer.top_k = original_top_k
        metadata["reranking"] = {"method": "standard"}
    else:
        # No reranking - raw results
        final_results = results[:request.top_k]
        metadata["reranking"] = {"skipped": True}

    metadata["rerank_time_ms"] = round((time.time() - rerank_start) * 1000, 2)

    # STEP 4: Format contexts
    contexts = []
    for idx, payload, score in final_results:
        meta = payload.get("metadata", {})
        context_dict = {
            "id": idx,
            "text": payload.get("text", ""),
            "doc_title": payload.get("doc_title", "Unknown"),
            "position": payload.get("position", 0),
            "retrieval_score": payload.get("retrieval_score"),
            "url": meta.get("url"),
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

    # STEP 5: Prompt Template Selection
    prompt_template = request.prompt_template

    if prompt_template == "auto":
        # Auto-select based on query analysis
        if is_multihop:
            prompt_template = "multihop"
        else:
            prompt_template = "enhanced"

    metadata["prompt"] = {"template": prompt_template}

    # STEP 6: Build Prompt
    prompt_builder = PromptBuilder()

    # Detect think style from model
    model_name = settings.llm.model_id.lower() if not request.provider or request.provider == settings.llm.provider else ""
    think_style = "qwen" if "qwen" in model_name else "none"

    config = PromptConfig(
        use_cot=request.use_cot,
        include_metadata=True,
        strict_citations=True,
        detect_language=True,
        think_tag_style=think_style,
    )

    if prompt_template == "multihop" and is_multihop:
        prompt, citation_mapping = prompt_builder.build(
            query=query,
            contexts=contexts,
            template_name="multihop",
            config=config,
            is_multihop=True,
            sub_queries=queries,
        )
    elif prompt_template == "enhanced":
        prompt_result = prompt_builder.build(
            query=query,
            contexts=contexts,
            template_name="enhanced",
            config=config,
        )
        if isinstance(prompt_result, tuple):
            prompt = prompt_result[0]
        else:
            prompt = prompt_result
    else:  # basic
        prompt_result = prompt_builder.build(
            query=query,
            contexts=contexts,
            template_name="basic",
            config=config,
        )
        if isinstance(prompt_result, tuple):
            prompt = prompt_result[0]
        else:
            prompt = prompt_result

    # STEP 7: LLM Generation
    llm_start = time.time()

    # Use custom provider if specified
    if request.provider and request.provider != settings.llm.provider:
        llm = LLMInference(provider=request.provider)
        metadata["llm"] = {"provider": request.provider, "override": True}
    else:
        llm = LLMInference()
        metadata["llm"] = {"provider": settings.llm.provider, "override": False}

    answer = llm.generate(
        prompt=prompt,
        chain_of_thought_enabled=request.use_cot,
    )

    metadata["llm_time_ms"] = round((time.time() - llm_start) * 1000, 2)

    # STEP 8: Citation Remapping (if multihop)
    if is_multihop and citation_mapping:
        answer, contexts = remap_citations(answer, citation_mapping, contexts)
        metadata["citation_remapping"] = {"applied": True}
    else:
        metadata["citation_remapping"] = {"applied": False}

    # STEP 9: CoVe Verification (optional)
    final_answer = answer

    if request.use_cove:
        cove_start = time.time()
        cove_result = cove_enhancer.verify(
            query=query,
            answer=answer,
            contexts=contexts,
        )

        if cove_result.corrected_answer:
            final_answer = cove_result.corrected_answer

        metadata["cove"] = {
            "status": str(cove_result.status),
            "needs_correction": cove_result.needs_correction,
            **cove_result.metadata,
        }
        metadata["cove_time_ms"] = round((time.time() - cove_start) * 1000, 2)
    else:
        metadata["cove"] = {"skipped": True}

    # Total time
    metadata["total_time_ms"] = round((time.time() - start) * 1000, 2)
    metadata["num_sources"] = len(contexts)

    logger.info(f"Ablation complete: total_time={metadata['total_time_ms']}ms")

    return PipelineAblationResponse(
        answer=final_answer,
        sources=contexts,
        metadata=metadata,
    )
