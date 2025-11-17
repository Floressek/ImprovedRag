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
        f"enhanced_features={request.enhanced_features_enabled}, cot={request.cot_enabled}, "
        f"reranking={request.reranker_enabled}, cove_mode={request.cove_mode}"
    )

    # Initialize metadata tracking
    metadata = {
        "ablation_config": {
            "query_analysis_enabled": request.query_analysis_enabled,
            "enhanced_features_enabled": request.enhanced_features_enabled,
            "cot_enabled": request.cot_enabled,
            "reranking_enabled": request.reranker_enabled,
            "cove_mode": request.cove_mode,
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

    # STAGE 2: Multihop Decomposition (auto-determined by query analysis)
    if request.query_analysis_enabled and features:
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
                vector=sq_embedding,
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
            vector=query_embedding,
            top_k=request.top_k * 2,  # Over-retrieve for reranking
        )
        results_by_subquery = {rewritten_query: initial_results}

    metadata["timings"]["retrieval_ms"] = (time.time() - start_retrieval) * 1000
    metadata["initial_results_count"] = len(initial_results)

    # Edge case: No results found
    if not initial_results:
        logger.warning("No results found from vector store - returning empty answer")
        total_time_ms = (time.time() - start_total) * 1000
        metadata["total_time_ms"] = total_time_ms
        metadata["final_results_count"] = 0
        metadata["sources_count"] = 0
        metadata["num_contexts"] = 0
        metadata["multihop_coverage"] = 0.0

        return {
            "answer": "I couldn't find any relevant information to answer your question.",
            "contexts": [],
            "context_details": [],
            "sub_queries": sub_queries,
            "sources": [],
            "metadata": metadata,
        }

    # STAGE 4: Reranking (optional)
    if request.reranker_enabled:
        start_rerank = time.time()

        if is_multihop:
            # Multihop reranking with diversity (3-stage: local → fusion → global)
            reranked_results = multihop_reranker.process(
                original_query=query,
                results_by_subquery=results_by_subquery,
                override_top_k=request.top_k,
                query_type=query_type,
            )
        else:
            # Standard reranking (single query)
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

    # Build contexts as List[Dict[str, Any]] (NOT List[str])
    contexts = []
    for doc_id, payload, score in final_results:
        contexts.append({
            "id": doc_id,
            "text": payload.get("text", ""),
            "doc_title": payload.get("doc_title", "Unknown"),
            "url": payload.get("url", ""),
            "retrieval_score": score,
        })

    # Build prompt
    provider = request.provider or settings.llm.provider
    llm = LLMInference(provider=provider)

    # Select template (Toggle 1: Query Analysis controls this)
    if request.prompt_template == "auto":
        if is_multihop:
            template = "multihop"  # Multihop detected → use multihop template
        else:
            template = "enhanced"  # Single query → use enhanced template (NOT basic)
    else:
        template = request.prompt_template

    # Build prompt config with Toggles 2 & 3
    prompt_config = PromptConfig(
        use_cot=request.cot_enabled,  # Toggle 3: Chain of Thought
        include_metadata=request.enhanced_features_enabled,  # Toggle 2: Enhanced Features
        strict_citations=request.enhanced_features_enabled,
        detect_language=request.enhanced_features_enabled,
        check_contradictions=request.enhanced_features_enabled,
        confidence_scoring=request.enhanced_features_enabled,
        think_tag_style="qwen" if request.cot_enabled else "none",  # CoT style
    )

    builder = PromptBuilder()

    # Build prompt with proper signature and handle citation_mapping
    citation_mapping = None
    if is_multihop and len(sub_queries) > 1:
        prompt, citation_mapping = builder.build(
            query=query,
            contexts=contexts,
            template_name=template,  # Use selected template
            config=prompt_config,  # Pass config here
            is_multihop=True,
            sub_queries=sub_queries,
        )
    else:
        prompt = builder.build(
            query=query,
            contexts=contexts,
            template_name=template,  # Use selected template
            config=prompt_config,  # Pass config here
            is_multihop=False,
            sub_queries=None,
        )

    # Generate answer (CoT already handled in prompt_config)
    answer = llm.generate(
        prompt=prompt,
        temperature=0.1,
        max_new_tokens=500,
    )

    metadata["timings"]["generation_ms"] = (time.time() - start_generation) * 1000
    metadata["prompt_template_used"] = template

    # STAGE 6: CoVe Verification (Toggle 5) - mode: off/auto/metadata/suggest
    cove_enabled = request.cove_mode != "off"

    if cove_enabled:
        start_cove = time.time()
        logger.info(f"Running CoVe in mode: {request.cove_mode}")

        # Prepare contexts for CoVe (full payloads with all metadata)
        cove_contexts = [payload for _, payload, _ in final_results]

        # Temporarily override cove settings for this request
        original_cove_enabled = settings.cove.enabled
        original_correction_mode = settings.cove.correction_mode

        settings.cove.enabled = True
        settings.cove.correction_mode = request.cove_mode  # "auto", "metadata", or "suggest"

        # Run CoVe verification
        cove_result = cove_enhancer.verify(
            query=query,
            answer=answer,
            contexts=cove_contexts,
        )

        # Restore original settings
        settings.cove.enabled = original_cove_enabled
        settings.cove.correction_mode = original_correction_mode

        # Update answer if corrections were made
        if cove_result.corrected_answer:
            answer = cove_result.corrected_answer
            metadata["cove_corrections_made"] = True
            metadata["cove_num_claims"] = cove_result.metadata.get("num_claims", 0)
            metadata["cove_num_refuted"] = cove_result.metadata.get("num_refuted", 0)
            metadata["cove_status"] = str(cove_result.status)
            metadata["cove_mode_used"] = request.cove_mode
        else:
            metadata["cove_corrections_made"] = False
            metadata["cove_num_claims"] = cove_result.metadata.get("num_claims", 0)
            metadata["cove_mode_used"] = request.cove_mode

        # Merge CoVe evidences into contexts (recovery sources)
        all_evidences = cove_result.metadata.get("all_evidences", [])
        if all_evidences:
            logger.info(f"Merging {len(all_evidences)} CoVe evidences into contexts")

            # Track existing doc IDs to avoid duplicates
            existing_ids = {doc_id for doc_id, _, _ in final_results}

            # Add new evidences that aren't already in contexts
            new_evidences_added = 0
            for ev in all_evidences:
                if ev["id"] not in existing_ids:
                    # Add to final_results as tuple (id, payload, score)
                    # Use "doc_title" key to match vector store payload structure
                    payload = {
                        "text": ev["text"],
                        "doc_title": ev.get("doc_title", "Unknown"),
                        "url": ev.get("url", ""),
                    }
                    final_results.append((ev["id"], payload, ev["score"]))
                    existing_ids.add(ev["id"])
                    new_evidences_added += 1

            if new_evidences_added > 0:
                logger.info(f"✓ Added {new_evidences_added} new evidences from CoVe recovery")
                metadata["cove_evidences_added"] = new_evidences_added

                # Rebuild contexts list to include new evidences (as List[Dict])
                contexts = []
                for doc_id, payload, score in final_results:
                    contexts.append({
                        "id": doc_id,
                        "text": payload.get("text", ""),
                        "doc_title": payload.get("doc_title", "Unknown"),
                        "url": payload.get("url", ""),
                        "retrieval_score": score,
                    })
            else:
                logger.debug("No new evidences added (all were already in contexts)")
                metadata["cove_evidences_added"] = 0
        else:
            metadata["cove_evidences_added"] = 0

        metadata["timings"]["cove_ms"] = (time.time() - start_cove) * 1000
    else:
        # CoVe mode = "off" - set all metrics to default/false
        metadata["cove_corrections_made"] = False
        metadata["cove_evidences_added"] = 0
        metadata["cove_num_claims"] = 0
        metadata["cove_mode_used"] = "off"
        logger.info("CoVe disabled (mode: off)")

    # STAGE 7: Remap citations (if needed)
    if citation_mapping:
        # Multihop: remap citations and reorganize contexts
        final_answer, contexts = remap_citations(answer, citation_mapping, contexts)
        logger.info("Remapped citations for multihop query")
    else:
        # Non-multihop: use answer as-is
        final_answer = answer

    # Calculate total time
    total_time_ms = (time.time() - start_total) * 1000
    metadata["total_time_ms"] = total_time_ms

    # Build context details from contexts (AFTER remap_citations)
    # This ensures citation_id alignment with remapped citations
    context_details = []
    for ctx in contexts:
        detail = {
            "text": ctx.get("text", ""),
            "url": ctx.get("url", ""),
            "title": ctx.get("doc_title", ctx.get("title", "")),
            "score": ctx.get("retrieval_score", 0.0),
        }
        # Add citation_id only if present (cited contexts)
        if "citation_id" in ctx:
            detail["citation_id"] = ctx["citation_id"]
        context_details.append(detail)

    # Extract unique sources from contexts (not context_details)
    sources = []
    seen_urls = set()
    for ctx in contexts:
        url = ctx.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "url": url,
                "title": ctx.get("doc_title", ctx.get("title", "")),
                "citation_id": ctx.get("citation_id"),  # May be None for uncited
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

    # Convert contexts to list[str] for schema compliance (RAGAS expects List[str])
    contexts_text = [ctx.get("text", "") for ctx in contexts]

    return {
        "answer": final_answer,
        "contexts": contexts_text,  # list[str] as per schema
        "context_details": context_details,  # Full details with citation_id, url, etc.
        "sub_queries": sub_queries,
        "sources": sources,
        "metadata": metadata,
    }
