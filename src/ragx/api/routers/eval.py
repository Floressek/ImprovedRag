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
    start_total = time.time()
    query = request.query

    logger.info(
        f"Ablation: query_analysis={request.query_analysis_enabled}, "
        f"multihop={request.multihop_enabled}, reranker={request.reranker_enabled}, "
        f"cove={request.cove_enabled}, template={request.prompt_template}"
    )

    # Initialize metadata tracking
    metadata = {
        "ablation_config": {
            "query_analysis_enabled": request.query_analysis_enabled,
            "multihop_enabled": request.multihop_enabled,
            "reranker_enabled": request.reranker_enabled,
            "cove_enabled": request.cove_enabled,
            "use_cot": request.use_cot,
            "prompt_template": request.prompt_template,
            "provider": request.provider or settings.llm.provider,
            "top_k": request.top_k,
        },
        "timings": {},
    }

    sub_queries = []
    query_type = "simple"
    is_multihop = False

    # STAGE 1: Query Analysis (optional)
    if request.query_analysis_enabled:
        start_analysis = time.time()
        features = linguistic_analyzer.analyze(query)
        metadata["linguistic_features"] = features.model_dump()
        metadata["timings"]["query_analysis_ms"] = (time.time() - start_analysis) * 1000

        # Optional: Query rewriting
        rewritten_query = adaptive_rewriter.rewrite(query, features)
        query_type = features.query_type or "simple"
        metadata["query_type"] = query_type
        metadata["rewritten_query"] = rewritten_query
    else:
        features = None
        rewritten_query = query
        metadata["query_type"] = "simple"

    # STAGE 2: Multihop Decomposition (optional)
    if request.multihop_enabled and request.query_analysis_enabled and features:
        # Check if query should be decomposed
        should_decompose = (
            features.query_type in ["comparison", "multihop"] or
            features.has_multiple_entities or
            features.has_conjunction
        )

        if should_decompose:
            start_decomp = time.time()
            from src.ragx.retrieval.rewriters.multihop_decomposer import MultihopDecomposer
            decomposer = MultihopDecomposer()
            decomposition = decomposer.decompose(rewritten_query, features)

            if decomposition and len(decomposition.sub_queries) > 1:
                sub_queries = decomposition.sub_queries
                is_multihop = True
                metadata["sub_queries"] = sub_queries
                metadata["is_multihop"] = True
                metadata["timings"]["decomposition_ms"] = (time.time() - start_decomp) * 1000
            else:
                sub_queries = [rewritten_query]
                metadata["is_multihop"] = False
        else:
            sub_queries = [rewritten_query]
            metadata["is_multihop"] = False
    else:
        sub_queries = [rewritten_query]
        metadata["is_multihop"] = False

    # STAGE 3: Retrieval
    start_retrieval = time.time()

    if is_multihop and len(sub_queries) > 1:
        # Multihop retrieval: retrieve for each sub-query
        results_by_subquery = {}
        all_results = []

        for sq in sub_queries:
            sq_embedding = embedder.embed_query(sq)
            sq_results = vector_store.search(
                query_vector=sq_embedding,
                top_k=request.top_k * 2,  # Over-retrieve for reranking
            )
            results_by_subquery[sq] = sq_results
            all_results.extend(sq_results)

        # Deduplicate
        seen = set()
        unique_results = []
        for doc_id, payload, score in all_results:
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append((doc_id, payload, score))

        initial_results = unique_results
        metadata["results_by_subquery_count"] = {sq: len(results_by_subquery[sq]) for sq in sub_queries}
    else:
        # Simple retrieval
        query_embedding = embedder.embed_query(rewritten_query)
        initial_results = vector_store.search(
            query_vector=query_embedding,
            top_k=request.top_k * 2,  # Over-retrieve for reranking
        )
        results_by_subquery = {rewritten_query: initial_results}

    metadata["timings"]["retrieval_ms"] = (time.time() - start_retrieval) * 1000
    metadata["initial_results_count"] = len(initial_results)

    # STAGE 4: Reranking (optional)
    if request.reranker_enabled:
        start_rerank = time.time()

        if is_multihop and request.multihop_enabled:
            # Multihop reranking with diversity
            reranked_results = multihop_reranker.process(
                original_query=query,
                results_by_subquery=results_by_subquery,
                override_top_k=request.top_k,
                query_type=query_type,
            )
        else:
            # Standard reranking
            reranked_results = reranker_enhancer.process(
                query=rewritten_query,
                results=initial_results,
            )

        final_results = reranked_results
        metadata["timings"]["reranking_ms"] = (time.time() - start_rerank) * 1000
    else:
        # No reranking: just take top_k
        final_results = initial_results[:request.top_k]

    metadata["final_results_count"] = len(final_results)

    # STAGE 5: Generation
    start_generation = time.time()

    # Build contexts
    contexts = [payload.get("text", "") for _, payload, _ in final_results]

    # Build prompt
    provider = request.provider or settings.llm.provider
    llm = LLMInference(provider=provider)

    # Select template
    if request.prompt_template == "auto":
        if is_multihop:
            template = "multihop"
        elif query_type == "comparison":
            template = "enhanced"
        else:
            template = "basic"
    else:
        template = request.prompt_template

    prompt_config = PromptConfig(
        template=template,
        include_context_citations=True,
        include_instructions=True,
    )

    builder = PromptBuilder(config=prompt_config)
    prompt = builder.build(
        query=query,
        contexts=contexts,
    )

    # Generate answer
    answer = llm.generate(
        prompt=prompt,
        temperature=0.1,
        max_new_tokens=500,
        chain_of_thought_enabled=request.use_cot,
    )

    metadata["timings"]["generation_ms"] = (time.time() - start_generation) * 1000
    metadata["prompt_template_used"] = template

    # STAGE 6: CoVe Verification (optional)
    if request.cove_enabled:
        start_cove = time.time()

        # Prepare contexts for CoVe (expects List[Dict[str, Any]])
        cove_contexts = [payload for _, payload, _ in final_results]

        # Run CoVe verification
        cove_result = cove_enhancer.verify(
            query=query,
            answer=answer,
            contexts=cove_contexts,
        )

        # Update answer if corrections were made
        if cove_result.corrected_answer:
            answer = cove_result.corrected_answer
            metadata["cove_corrections_made"] = True
            metadata["cove_num_corrections"] = len(cove_result.corrections)
        else:
            metadata["cove_corrections_made"] = False

        metadata["timings"]["cove_ms"] = (time.time() - start_cove) * 1000
    else:
        metadata["cove_corrections_made"] = False

    # STAGE 7: Remap citations (if needed)
    # Extract citations from contexts for answer
    final_answer = remap_citations(answer, contexts)

    # Calculate total time
    total_time_ms = (time.time() - start_total) * 1000
    metadata["total_time_ms"] = total_time_ms

    # Build context details for response
    context_details = []
    for i, (doc_id, payload, score) in enumerate(final_results):
        context_details.append({
            "citation_id": i + 1,
            "text": payload.get("text", ""),
            "url": payload.get("url", ""),
            "title": payload.get("title", ""),
            "score": score,
        })

    # Extract unique sources
    sources = []
    seen_urls = set()
    for ctx in context_details:
        url = ctx.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "url": url,
                "title": ctx.get("title", ""),
                "citation_id": ctx.get("citation_id"),
            })

    # Add custom metrics
    metadata["sources_count"] = len(sources)
    metadata["num_contexts"] = len(contexts)

    # Calculate multihop coverage
    if is_multihop and len(sub_queries) > 1:
        covered = sum(1 for sq in sub_queries if results_by_subquery.get(sq, []))
        metadata["multihop_coverage"] = covered / len(sub_queries)
    else:
        metadata["multihop_coverage"] = 1.0

    return {
        "answer": final_answer,
        "contexts": contexts,
        "context_details": context_details,
        "sub_queries": sub_queries,
        "sources": sources,
        "metadata": metadata,
    }
