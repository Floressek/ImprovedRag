from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Tuple, Optional

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
        reranker_enabled: bool,
) -> Tuple[List[Any], Dict[str, List[Any]], int]:
    """Perform retrieval for single or multihop queries."""
    if reranker_enabled:
        retrieve_k = settings.retrieval.top_k_retrieve  # np. 200 from config
    else:
        retrieve_k = top_k * 2
    if is_multihop and len(queries) > 1:
        results_by_subquery = {}
        for sub_query in queries:
            qvec = embedder.embed_query(sub_query)
            results = vector_store.search(
                vector=qvec,
                top_k=retrieve_k,
                hnsw_ef=settings.hnsw.search_ef
            )
            results_by_subquery[sub_query] = results

        num_retrieved = sum(len(v) for v in results_by_subquery.values())
        return [], results_by_subquery, num_retrieved
    else:
        query_singular = queries[0]
        qvec = embedder.embed_query(query_singular)
        results = vector_store.search(
            vector=qvec,
            top_k=retrieve_k,
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
        results = reranker_enhancer.process(original_query, all_results, top_k)

        reranker_enhancer.top_k = original_top_k
        return results


def _format_contexts(results: List[Any], is_multihop: bool) -> List[Dict[str, Any]]:
    """Format retrieval results as context dicts."""
    contexts = []
    for idx, payload, score in results:
        payload_metadata = payload.get("metadata", {})
        retrieval_score = payload.get("retrieval_score")
        if retrieval_score is None:
            retrieval_score = float(score)
        context_dict = {
            "id": idx,
            "text": payload.get("text", ""),
            "doc_title": payload.get("doc_title", "Unknown"),
            "position": payload.get("position", 0),
            "total_chunks": payload.get("total_chunks", 1),
            "retrieval_score": retrieval_score,
            "url": payload_metadata.get("url"),
        }

        if is_multihop:
            context_dict["local_rerank_score"] = payload.get("local_rerank_score") or 0.0
            context_dict["fused_score"] = payload.get("fused_score") or 0.0
            context_dict["global_rerank_score"] = payload.get("global_rerank_score") or 0.0
            context_dict["final_score"] = payload.get("final_score") or 0.0
            context_dict["fusion_metadata"] = payload.get("fusion_metadata")

            fusion_meta = payload.get("fusion_metadata", {})
            source_subqueries = fusion_meta.get("source_subqueries", [])
            if source_subqueries:
                context_dict["source_subquery"] = source_subqueries[0]
        else:
            if payload.get("rerank_score") is not None:
                context_dict["rerank_score"] = payload.get("rerank_score")

        contexts.append(context_dict)

    return contexts


def _select_template(
        prompt_template: str,
        is_multihop: bool,
        query_analysis_enabled: bool,
) -> str:
    """Select appropriate template based on configuration."""
    if is_multihop:
        return "multihop"

    if prompt_template in ["basic", "enhanced"]:
        return prompt_template

    # Auto: query_analysis ON = enhanced, OFF = basic
    if prompt_template == "auto":
        return "enhanced" if query_analysis_enabled else "basic"

    return "enhanced"


def _generate_answer(
        query: str,
        queries: List[str],
        contexts: List[Dict[str, Any]],
        is_multihop: bool,
        cot_enabled: bool,
        query_analysis_enabled: bool,
        prompt_template: str,
        provider: str,
) -> Tuple[str, Any, str]:
    """Generate answer using LLM.

    Returns:
        Tuple of (answer, citation_mapping, template_used)
    """
    # Select template
    template_name = _select_template(prompt_template, is_multihop, query_analysis_enabled)

    # Enhanced features are enabled for enhanced/multihop templates
    use_enhanced = template_name in ["enhanced", "multihop"]

    prompt_config = PromptConfig(
        use_cot=cot_enabled,
        include_metadata=use_enhanced,
        strict_citations=use_enhanced,
        detect_language=use_enhanced,
        check_contradictions=use_enhanced,
        confidence_scoring=use_enhanced,
        think_tag_style="qwen" if cot_enabled else "none",
    )

    llm = LLMInference(provider=provider)
    prompt_builder = PromptBuilder()

    if is_multihop:
        prompt, citation_mapping = prompt_builder.build(
            query=query,
            contexts=contexts,
            template_name=template_name,
            config=prompt_config,
            is_multihop=True,
            sub_queries=queries,
        )
    else:
        prompt = prompt_builder.build(
            query=query,
            contexts=contexts,
            template_name=template_name,
            config=prompt_config,
            is_multihop=False,
            sub_queries=None,
        )
        citation_mapping = None

    # answer = llm.generate(prompt, temperature=0.1, max_new_tokens=500)
    answer = llm.generate(
        prompt=prompt,
        chain_of_thought_enabled=cot_enabled
    )
    return answer, citation_mapping, template_name


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

    # Merge evidences - ONLY verified claims (supports), NOT refuted
    # IMPORTANT: For RAGAS evaluation, we MUST filter evidences by verification_label
    # to avoid adding refuted/contradictory evidences that would lower faithfulness scores
    all_evidences = cove_result.metadata.get("all_evidences", [])
    if all_evidences:
        original_context_ids = {ctx.get("id") for ctx in contexts if ctx.get("id") is not None}
        existing_ids = {ctx.get("id") for ctx in sources if ctx.get("id") is not None}

        for ev in all_evidences:
            ev_id = ev.get("id")
            if not ev_id or ev_id in existing_ids or str(ev_id).startswith("unknown_"):
                continue

            # Filter 1: only verified claims (label="supports")
            verification_label = ev.get("verification_label", "unknown")
            if verification_label != "supports":
                logger.debug(f"Skipping CoVe evidence with label '{verification_label}': {ev_id}")
                continue

            # Filter 2: only evidences from ORIGINAL contexts (skip recovery evidences)
            # Recovery evidences have new doc_ids that weren't in original retrieval
            if ev_id not in original_context_ids:
                logger.debug(f"Skipping CoVe recovery evidence (not in original contexts): {ev_id}")
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


def _validate_config(
        query_analysis_enabled: bool,
        reranker_enabled: bool,
        cove_mode: str,
        prompt_template: str,
        top_k: int,
        is_multihop: bool,
        num_queries: int,
) -> Optional[Dict[str, Any]]:
    """
    Validate ablation configuration for incompatible toggle combinations.

    Args:
        query_analysis_enabled: Whether query analysis is enabled
        reranker_enabled: Whether reranker is enabled
        is_multihop: Whether query was detected as multihop
        num_queries: Number of sub-queries

    Returns:
        Error message if invalid, None if valid
    """
    warnings = []
    # Rule 1: Multihop requires reranker (3-stage: local→fusion→global)
    if is_multihop and num_queries > 1 and not reranker_enabled:
        return {
            "error": "Invalid Configuration",
            "message": (
                "Multihop queries require reranker to be enabled. "
                "The multihop flow uses 3-stage reranking (local→fusion→global) which is "
                "an integral part of the pipeline and cannot be disabled separately."
            ),
            "detected": {
                "is_multihop": True,
                "num_sub_queries": num_queries,
                "reranker_enabled": False,
            },
            "solutions": [
                "Set 'use_reranker': true to enable multihop flow",
                "Set 'use_query_analysis': false to force single-query mode",
            ],
        }

    if cove_mode != "off" and top_k == 0:
        return {
            "error": "Invalid Configuration",
            "message": "CoVe verification requires contexts (top_k > 0)",
            "detected": {
                "cove_mode": cove_mode,
                "top_k": top_k,
            },
            "solutions": [
                "Set 'top_k' > 0 to retrieve contexts for CoVe",
                "Set 'cove': 'off' to disable verification",
            ],
        }

    # Warning 1: Multihop ignores prompt_template
    if is_multihop and prompt_template == "basic":
        warnings.append({
            "type": "ignored_parameter",
            "message": (
                "prompt_template='basic' is IGNORED for multihop queries. "
                "Multihop always uses 'multihop' template (hardcoded)."
            ),
            "actual_template": "multihop",
        })

    # Warning 2: Query analysis OFF but multihop prompt requested
    if not query_analysis_enabled and prompt_template == "multihop":
        warnings.append({
            "type": "impossible_request",
            "message": (
                "You requested prompt_template='multihop' but 'use_query_analysis' is disabled. "
                "Without query analysis, there are no sub_queries, so 'enhanced' template will be used instead."
            ),
            "actual_template": "enhanced",
        })

    # Return warnings if any (non-blocking)
    if warnings:
        return {
            "status": "warning",
            "warnings": warnings,
            "message": "Configuration has warnings but will proceed",
        }

    return None


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

    4 Independent Toggles:
    1. query_analysis_enabled: Enable/disable adaptive_rewriter
    2. cot_enabled: Enable/disable Chain-of-Thought
    3. reranker_enabled: Enable/disable reranking
    4. cove_mode: "off", "auto", "metadata", "suggest"

    Plus prompt_template selection: "basic", "enhanced", "auto"
    """
    start_total = time.time()

    # Initialize metadata
    metadata: Dict[str, Any] = {
        "ablation_config": {
            "query_analysis_enabled": request.query_analysis_enabled,
            "cot_enabled": request.cot_enabled,
            "reranker_enabled": request.reranker_enabled,
            "cove_mode": request.cove_mode,
            "prompt_template": request.prompt_template,
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

    validation_result = _validate_config(
        query_analysis_enabled=request.query_analysis_enabled,
        reranker_enabled=request.reranker_enabled,
        cove_mode=request.cove_mode,
        prompt_template=request.prompt_template,
        top_k=request.top_k,
        is_multihop=is_multihop,
        num_queries=len(queries),
    )
    if validation_result:
        if validation_result.get("error"):
            # Hard error - 400
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=validation_result)
        elif validation_result.get("warnings"):
            logger.warning(f"Config warnings: {validation_result['warnings']}")
            metadata["config_warnings"] = validation_result["warnings"]

    # STEP 2: Retrieval
    retrieval_start = time.time()
    results, results_by_subquery, num_retrieved = _perform_retrieval(
        queries, is_multihop, request.top_k, embedder, vector_store, request.reranker_enabled
    )
    retrieval_time_ms = round((time.time() - retrieval_start) * 1000, 2)
    metadata["timings"]["retrieval_time_ms"] = retrieval_time_ms
    metadata["num_candidates"] = num_retrieved

    # Edge case: No results
    if num_retrieved == 0:
        total_time_ms = round((time.time() - start_total) * 1000, 2)

        # Metadata in 1:1 format as in EnhancedPipeline + extras from ablation
        metadata["pipeline"] = "ablation"
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
        metadata["template_used"] = "none"

        return PipelineAblationResponse(
            answer="I couldn't find any relevant information to answer your question.",
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
    answer, citation_mapping, template_used = _generate_answer(
        request.query,
        queries,
        contexts,
        is_multihop,
        request.cot_enabled,
        request.query_analysis_enabled,
        request.prompt_template,
        provider,
    )
    llm_time_ms = round((time.time() - llm_start) * 1000, 2)
    metadata["timings"]["llm_time_ms"] = llm_time_ms
    metadata["template_used"] = template_used

    # Remap citations (multihop only)
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

    metadata["pipeline"] = "ablation"
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

    # Add results_by_subquery for multihop coverage calculation
    # Format: {sub_query: num_results} - simplified, only need count for coverage
    if results_by_subquery:
        metadata["results_by_subquery"] = {
            sq: len(sq_results)
            for sq, sq_results in results_by_subquery.items()
        }

    # KEY CHANGE:
    # sources: return WITHOUT processing, exactly as in EnhancedPipeline.answer
    # (full list of contexts + evidences from CoVe)
    return PipelineAblationResponse(
        answer=final_answer,
        sources=sources,
        metadata=metadata,
    )
