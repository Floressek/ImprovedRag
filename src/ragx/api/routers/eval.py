from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Tuple

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


def _perform_query_analysis(
        query: str,
        query_analysis_enabled: bool,
        adaptive_rewriter: AdaptiveQueryRewriter,
) -> Tuple[str, List[str], bool, str, str]:
    """Perform query analysis and rewriting.

    Returns:
        original_query, queries, is_multihop, query_type, reasoning
    """
    if query_analysis_enabled:
        rewrite_result = adaptive_rewriter.rewrite(query)
        original_query = rewrite_result["original"]
        queries = rewrite_result["queries"]
        is_multihop = rewrite_result["is_multihop"]
        query_type = rewrite_result.get("query_type", "general")
        reasoning = rewrite_result.get("reasoning", "")
    else:
        original_query = query
        queries = [query]
        is_multihop = False
        query_type = "general"
        reasoning = "query_analysis_disabled"

    return original_query, queries, is_multihop, query_type, reasoning


def _perform_retrieval(
        queries: List[str],
        is_multihop: bool,
        top_k: int,
        embedder: Embedder,
        vector_store: QdrantStore,
) -> Tuple[List[Any], Dict[str, List[Any]], int]:
    """Perform retrieval for single or multihop queries."""
    if is_multihop and len(queries) > 1:
        results_by_subquery = {}
        for sub_query in queries:
            qvec = embedder.embed_query(sub_query)
            results = vector_store.search(
                vector=qvec,
                top_k=top_k * 2,
                hnsw_ef=settings.hnsw.search_ef
            )
            results_by_subquery[sub_query] = results

        num_retrieved = sum(len(v) for v in results_by_subquery.values())
        all_results = []
        return all_results, results_by_subquery, num_retrieved
    else:
        query_singular = queries[0]
        qvec = embedder.embed_query(query_singular)
        results = vector_store.search(
            vector=qvec,
            top_k=top_k * 2,
            hnsw_ef=settings.hnsw.search_ef
        )
        results_by_subquery = {query_singular: results}
        return results, results_by_subquery, len(results)


def _perform_reranking(
        reranker_enabled: bool,
        is_multihop: bool,
        queries: List[str],
        original_query: str,
        query_type: str,
        top_k: int,
        results_by_subquery: Dict[str, List[Any]],
        reranker_enhancer: RerankerEnhancer,
        multihop_reranker: MultihopRerankerEnhancer,
) -> List[Any]:
    """Perform reranking step."""
    if not reranker_enabled:
        query_for_results = queries[0] if queries else original_query
        all_results = results_by_subquery.get(query_for_results, [])
        return all_results[:top_k]

    if is_multihop and len(queries) > 1:
        return multihop_reranker.process(
            original_query=original_query,
            results_by_subquery=results_by_subquery,
            override_top_k=top_k,
            query_type=query_type
        )
    else:
        original_top_k = reranker_enhancer.top_k
        reranker_enhancer.top_k = top_k

        query_for_rerank = queries[0] if queries else original_query
        all_results = results_by_subquery.get(query_for_rerank, [])
        results = reranker_enhancer.process(original_query, all_results)

        reranker_enhancer.top_k = original_top_k
        return results


def _format_contexts(results: List[Any], is_multihop: bool) -> List[Dict[str, Any]]:
    """Format retrieval results as context dicts."""
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

    return contexts


def _generate_answer(
        query: str,
        queries: List[str],
        contexts: List[Dict[str, Any]],
        is_multihop: bool,
        cot_enabled: bool,
        enhanced_features_enabled: bool,
        provider: str,
) -> Tuple[str, Any]:
    """Generate answer using LLM."""
    prompt_config = PromptConfig(
        use_cot=cot_enabled,
        include_metadata=enhanced_features_enabled,
        strict_citations=enhanced_features_enabled,
        detect_language=enhanced_features_enabled,
        check_contradictions=enhanced_features_enabled,
        confidence_scoring=enhanced_features_enabled,
        think_tag_style="qwen" if cot_enabled else "none",
    )

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

    # answer = llm.generate(prompt, temperature=0.1, max_new_tokens=500)
    answer = llm.generate(prompt)
    return answer, citation_mapping


def _apply_cove(
        cove_mode: str,
        query: str,
        answer: str,
        contexts: List[Dict[str, Any]],
        cove_enhancer: CoVeEnhancer,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Apply CoVe verification if enabled."""
    sources = contexts.copy()
    cove_metadata = {}

    if cove_mode == "off":
        return answer, sources, cove_metadata

    original_cove_enabled = settings.cove.enabled
    settings.cove.enabled = True

    cove_result = cove_enhancer.verify(
        query=query,
        answer=answer,
        contexts=contexts,
        correction_mode=cove_mode,
    )

    settings.cove.enabled = original_cove_enabled

    final_answer = cove_result.corrected_answer if cove_result.corrected_answer else answer

    cove_metadata = {
        "status": str(cove_result.status),
        "needs_correction": cove_result.needs_correction,
        **cove_result.metadata,
    }

    # Merge evidences
    all_evidences = cove_result.metadata.get("all_evidences", [])
    if all_evidences:
        existing_ids = {ctx.get("id") for ctx in sources if ctx.get("id") is not None}

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

    return final_answer, sources, cove_metadata


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
    1. query_analysis_enabled: Enable/disable adaptive_rewriter
    2. enhanced_features_enabled: Enable/disable metadata in prompt
    3. cot_enabled: Enable/disable Chain-of-Thought
    4. reranker_enabled: Enable/disable reranking
    5. cove_mode: "off", "auto", "metadata", "suggest"
    """
    start_total = time.time()

    # Initialize metadata
    metadata: Dict[str, Any] = {
        "ablation_config": {
            "query_analysis_enabled": request.query_analysis_enabled,  # type of query / query rewrite + multihop
            "enhanced_features_enabled": request.enhanced_features_enabled,  # better prompt?
            "cot_enabled": request.cot_enabled,  # on/off CoT works
            "reranker_enabled": request.reranker_enabled,  # on/off reranker errors
            "cove_mode": request.cove_mode,  # works
        },
        "timings": {},
    }

    # STEP 1: Query Analysis (as in EnhancedPipeline + reasoning)
    rewrite_start = time.time()
    (
        original_query,
        queries,
        is_multihop,
        query_type,
        reasoning,
    ) = _perform_query_analysis(
        request.query,
        request.query_analysis_enabled,
        adaptive_rewriter,
    )
    rewrite_time_ms = round((time.time() - rewrite_start) * 1000, 2)
    metadata["timings"]["rewrite_time_ms"] = rewrite_time_ms
    metadata["is_multihop"] = is_multihop
    metadata["sub_queries"] = queries
    metadata["query_type"] = query_type
    metadata["reasoning"] = reasoning

    # STEP 2: Retrieval
    retrieval_start = time.time()
    results, results_by_subquery, num_retrieved = _perform_retrieval(
        queries, is_multihop, request.top_k, embedder, vector_store
    )
    retrieval_time_ms = round((time.time() - retrieval_start) * 1000, 2)
    metadata["timings"]["retrieval_time_ms"] = retrieval_time_ms
    metadata["num_candidates"] = num_retrieved

    # Edge case: No results
    if num_retrieved == 0:
        total_time_ms = round((time.time() - start_total) * 1000, 2)

        # Metadata in 1:1 format as in EnhancedPipeline + extras from ablation
        metadata["pipeline"] = "enhanced"
        metadata["phases"] = [
            "linguistic_analysis" if request.query_analysis_enabled else None,
            "adaptive_rewriting" if request.query_analysis_enabled else None,
            "retrieval",
            "reranking" if request.reranker_enabled else None,
            "generation",
            "cove_verification" if request.cove_mode != "off" else None,
        ]
        metadata["rewrite_time_ms"] = rewrite_time_ms
        metadata["retrieval_time_ms"] = retrieval_time_ms
        metadata["rerank_time_ms"] = 0.0
        metadata["llm_time_ms"] = 0.0
        metadata["cove_time_ms"] = 0.0
        metadata["total_time_ms"] = total_time_ms
        metadata["num_sources"] = 0
        metadata["timings"]["total_time_ms"] = total_time_ms

        return PipelineAblationResponse(
            answer="I couldn't find any relevant information to answer your question.",
            # contexts=[],
            # context_details=[],
            # sub_queries=queries,
            sources=[],
            metadata=metadata,
        )

    # STEP 3: Reranking
    rerank_start = time.time()
    results = _perform_reranking(
        request.reranker_enabled,
        is_multihop,
        queries,
        original_query,
        query_type,
        request.top_k,
        results_by_subquery,
        reranker_enhancer,
        multihop_reranker,
    )
    rerank_time_ms = round((time.time() - rerank_start) * 1000, 2)
    metadata["timings"]["rerank_time_ms"] = rerank_time_ms

    # STEP 4: Format Contexts (same shape as in EnhancedPipeline)
    contexts = _format_contexts(results, is_multihop)

    # STEP 5: Generation
    llm_start = time.time()
    provider = request.provider or settings.llm.provider
    answer, citation_mapping = _generate_answer(
        request.query,
        queries,
        contexts,
        is_multihop,
        request.cot_enabled,
        request.enhanced_features_enabled,
        provider,
    )
    llm_time_ms = round((time.time() - llm_start) * 1000, 2)
    metadata["timings"]["llm_time_ms"] = llm_time_ms

    # Remap citations (as in EnhancedPipeline)
    if is_multihop and citation_mapping:
        answer, contexts = remap_citations(answer, citation_mapping, contexts)

    # STEP 6: CoVe – behavior 1:1 as in EnhancedPipeline, just controlled by cove_mode
    cove_start = time.time()
    final_answer, sources, cove_metadata = _apply_cove(
        request.cove_mode,
        request.query,
        answer,
        contexts,
        cove_enhancer,
    )
    cove_time_ms = round((time.time() - cove_start) * 1000, 2)
    metadata["timings"]["cove_time_ms"] = cove_time_ms

    if cove_metadata:
        # Structure identical to EnhancedPipeline.metadata["cove"]
        metadata["cove"] = cove_metadata

    # Final metadata – 1:1 as in EnhancedPipeline + extra
    total_time_ms = round((time.time() - start_total) * 1000, 2)
    metadata["timings"]["total_time_ms"] = total_time_ms

    # Main fields same as in EnhancedPipeline.answer
    metadata["pipeline"] = "enhanced"  # exactly the same pipeline tag
    metadata["is_multihop"] = is_multihop
    metadata["sub_queries"] = queries if is_multihop else None
    metadata["phases"] = [
        "linguistic_analysis" if request.query_analysis_enabled else None,
        "adaptive_rewriting" if request.query_analysis_enabled else None,
        "retrieval",
        "multihop_reranking (local→fusion→global)" if (request.reranker_enabled and is_multihop)
        else ("reranking" if request.reranker_enabled else None),
        "generation",
        "cove_verification" if request.cove_mode != "off" else None,
    ]
    metadata["rewrite_time_ms"] = rewrite_time_ms
    metadata["retrieval_time_ms"] = retrieval_time_ms
    metadata["rerank_time_ms"] = rerank_time_ms
    metadata["llm_time_ms"] = llm_time_ms
    metadata["cove_time_ms"] = cove_time_ms
    metadata["total_time_ms"] = total_time_ms
    metadata["num_candidates"] = num_retrieved
    metadata["num_sources"] = len(sources)
    metadata["query_type"] = query_type
    metadata["reasoning"] = reasoning

    # Build response – EXTRA fields from ablation (contexts, context_details) stay
    contexts_text = [ctx.get("text", "") for ctx in contexts]

    context_details = []
    for ctx in contexts:
        detail = {
            "text": ctx.get("text", ""),
            "url": ctx.get("url", ""),
            "title": ctx.get("doc_title", ""),
            "score": ctx.get("rerank_score")
                     or ctx.get("final_score")
                     or ctx.get("retrieval_score", 0.0),
        }
        if "citation_id" in ctx:
            detail["citation_id"] = ctx["citation_id"]
        context_details.append(detail)

    # KEY CHANGE:
    # sources: return WITHOUT processing, exactly as in EnhancedPipeline.answer
    # (full list of contexts + evidences from CoVe)
    return PipelineAblationResponse(
        answer=final_answer,
        # contexts=contexts_text,            # EXTRA compared to enhanced pipeline
        # context_details=context_details,   # EXTRA
        # sub_queries=queries,
        sources=sources,                   # 1:1 as EnhancedPipeline.answer['sources']
        metadata=metadata,                 # 1:1 + ablation_config, timings, query_type
    )
