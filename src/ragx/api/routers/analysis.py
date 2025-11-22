from __future__ import annotations

import logging
import time
from fastapi import APIRouter, Depends

from src.ragx.api.dependencies import (
    get_linguistic_analyzer,
    get_adaptive_rewriter,
    get_embedder,
    get_vector_store,
    get_multihop_reranker, get_reranker_enhancer,
)
from src.ragx.api.schemas import LinguisticAnalysisResponse, LinguisticAnalysisRequest, MultihopSearchResponse, \
    MultihopSearchRequest, MultihopSearchResult
from src.ragx.api.schemas.analysis import RewrittenQuery
from src.ragx.pipelines.enhancers.reranker import RerankerEnhancer
from src.ragx.retrieval.analyzers.linguistic_analyzer import LinguisticAnalyzer
from src.ragx.retrieval.constants import QueryType
from src.ragx.retrieval.rewriters.adaptive_rewriter import AdaptiveQueryRewriter
from src.ragx.retrieval.embedder.embedder import Embedder
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore
from src.ragx.pipelines.enhancers.multihop_reranker import MultihopRerankerEnhancer
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("/linguistic", response_model=LinguisticAnalysisResponse)
async def analyze_query_linguistics(
        request: LinguisticAnalysisRequest,
        linguistic_analyzer: LinguisticAnalyzer = Depends(get_linguistic_analyzer),
) -> LinguisticAnalysisResponse:
    """
    Analyze linguistic features of a query.

    Returns detailed linguistic analysis including:
    - POS (Part-of-Speech) tags
    - Dependency tree
    - Named entities
    - Syntax complexity metrics
    """
    logger.info(f"Analyzing query: {request.query[:50]}")

    features = linguistic_analyzer.analyze(request.query)

    dep_tree_formatted = [
        {"dependency": dep, "head": head, "child": child}
        for dep, head, child in features.dep_tree
    ]

    entities_formatted = [
        {"text": text, "label": label}
        for text, label in features.entities
    ]

    return LinguisticAnalysisResponse(
        query=features.query,
        pos_sequence=features.pos_sequence,
        dep_tree=dep_tree_formatted,
        entities=entities_formatted,
        num_tokens=features.num_tokens,
        num_clauses=features.num_clauses,
        syntax_depth=features.syntax_depth,
        has_relative_clauses=features.has_relative_clauses,
        has_conjunctions=features.has_conjunctions,
        analysis_text=features.to_context_string(),
    )


@router.post("/multihop", response_model=MultihopSearchResponse)
async def search_multihop(
        request: MultihopSearchRequest,
        embedder: Embedder = Depends(get_embedder),
        vector_store: QdrantStore = Depends(get_vector_store),
        linguistic_analyzer: LinguisticAnalyzer = Depends(get_linguistic_analyzer),
        adaptive_rewriter: AdaptiveQueryRewriter = Depends(get_adaptive_rewriter),
        multihop_reranker: MultihopRerankerEnhancer = Depends(get_multihop_reranker),
        reranker_enhancer: RerankerEnhancer = Depends(get_reranker_enhancer)
) -> MultihopSearchResponse:
    """
    Multihop search with optional reranking and linguistic analysis.

    Features:
    - Automatic query decomposition into sub-queries
    - Parallel retrieval for each sub-query
    - Optional three-stage reranking (local → fusion → global)
    - Optional linguistic analysis

    Parameters:
    - use_reranker: Enable/disable reranking (if False, only fusion is applied)
    - include_linguistic_analysis: Include linguistic analysis in response
    """
    start_time = time.time()

    # Step 1: Query Rewriting
    rewrite_start = time.time()
    rewrite_result = adaptive_rewriter.rewrite(request.query)
    rewrite_time = (time.time() - rewrite_start) * 1000

    original_query = rewrite_result["original"]
    queries = rewrite_result["queries"]
    is_multihop = rewrite_result["is_multihop"]

    logger.info(
        f"Query analysis: multihop={is_multihop}, "
        f"queries={len(queries)}, reason={rewrite_result['reasoning']}"
    )

    # Step 2: Retrieval (parallel for multihop)
    retrieval_start = time.time()
    processing_method = ""
    total_retrieved = 0

    if is_multihop and len(queries) > 1:
        # Multihop: retrieve for each sub-query
        results_by_subquery = {}
        for sub_query in queries:
            qvec = embedder.embed_query(sub_query)
            results = vector_store.search(
                vector=qvec,
                top_k=request.top_k,
                hnsw_ef=settings.hnsw.search_ef
            )
            results_by_subquery[sub_query] = results
            logger.info(f"Retrieved {len(results)} for sub-query: {sub_query[:50]}...")

        retrieval_time = (time.time() - retrieval_start) * 1000

        total_retrieved = sum(len(v) for v in results_by_subquery.values())
        logger.info(f"Retrieved {total_retrieved} total candidates from {len(queries)} sub-queries")

        # Step 3: Multihop reranking (local → fusion → global)
        # NOTE: Fusion is handled INSIDE multihop_reranker
        rerank_start = time.time()

        if request.use_reranker:
            # Full three-stage reranking
            final_results = multihop_reranker.process(
                original_query=original_query,
                results_by_subquery=results_by_subquery,
                override_top_k=request.top_k,
            )
            processing_method = "three-stage reranking (local→fusion→global)"
        else:
            # Only local rerank + fusion (no global reranking)
            local_reranked = multihop_reranker._local_rerank(results_by_subquery)
            final_results = multihop_reranker._fuse_results(local_reranked)
            final_results = final_results[:request.top_k]
            processing_method = "local rerank + fusion (no global reranking)"

        rerank_time = (time.time() - rerank_start) * 1000
        logger.info(f"Processed {len(final_results)} candidates - multihop.")

    else:
        query_singular = queries[0] if queries else original_query
        qvec = embedder.embed_query(query_singular)
        results = vector_store.search(
            vector=qvec,
            top_k=request.top_k,
            hnsw_ef=settings.hnsw.search_ef
        )
        retrieval_time = (time.time() - retrieval_start) * 1000
        logger.info(f"Retrieved {len(results)} candidates - single query.")
        total_retrieved = len(results)

        # Step 3: Standard reranking
        rerank_start = time.time()

        if request.use_reranker:
            # Save original top_k
            original_reranker_top_k = reranker_enhancer.top_k
            reranker_enhancer.top_k = request.top_k
            logger.info(f"Using custom top_k = {request.top_k}")

            final_results = reranker_enhancer.process(original_query, results)
            reranker_enhancer.top_k = original_reranker_top_k
            processing_method = "standard reranking"
        else:
            # No reranking, just return top_k results
            final_results = results[:request.top_k]
            processing_method = "retrieval only (no reranking)"

        rerank_time = (time.time() - rerank_start) * 1000
        logger.info(f"Processed {len(final_results)} candidates")

    # Step 4: Format Results
    output = []
    for doc_id, payload, score in final_results:
        metadata = payload.get("metadata", {})

        result = MultihopSearchResult(
            id=str(doc_id),
            doc_title=payload.get("doc_title", "Unknown"),
            text=payload.get("text", ""),
            retrieval_score=payload.get("retrieval_score"),
            local_rerank_score=payload.get("local_rerank_score"),
            fused_score=payload.get("fused_score"),
            global_rerank_score=payload.get("global_rerank_score") if request.use_reranker else None,
            final_score=float(score),
            position=payload.get("position", 0),
            url=metadata.get("url"),
            fusion_metadata=payload.get("fusion_metadata"),
        )
        output.append(result)

    # Step 5: Optional linguistic analysis
    linguistic_response = None
    if request.include_linguistic_analysis:
        features = linguistic_analyzer.analyze(request.query)

        dep_tree_formatted = [
            {"dependency": dep, "head": head, "child": child}
            for dep, head, child in features.dep_tree
        ]

        entities_formatted = [
            {"text": text, "label": label}
            for text, label in features.entities
        ]

        linguistic_response = LinguisticAnalysisResponse(
            query=features.query,
            pos_sequence=features.pos_sequence,
            dep_tree=dep_tree_formatted,
            entities=entities_formatted,
            num_tokens=features.num_tokens,
            num_clauses=features.num_clauses,
            syntax_depth=features.syntax_depth,
            has_relative_clauses=features.has_relative_clauses,
            has_conjunctions=features.has_conjunctions,
            analysis_text=features.to_context_string(),
        )

    total_time = (time.time() - start_time) * 1000

    return MultihopSearchResponse(
        original_query=original_query,
        sub_queries=queries,
        results=output,
        linguistic_analysis=linguistic_response,
        metadata={
            "processing_method": processing_method,
            "use_reranker": request.use_reranker,
            "is_multihop": is_multihop,
            "num_sub_queries": len(queries),
            "total_retrieved": total_retrieved,
            "num_results": len(output),
            "rewrite_time_ms": round(rewrite_time, 2),
            "retrieval_time_ms": round(retrieval_time, 2),
            "processing_time_ms": round(rerank_time, 2),
            "total_time_ms": round(total_time, 2),
            "reasoning": rewrite_result["reasoning"],
        },
    )


@router.post("/rewrite", response_model=RewrittenQuery)
async def query_rewrite(
        request: LinguisticAnalysisRequest,
        adaptive_rewriter: AdaptiveQueryRewriter = Depends(get_adaptive_rewriter),
) -> RewrittenQuery:
    """
    Analyze and rewrite query using adaptive query rewriter.
    Returns:
    - Original query
    - Queries (sub-queries if multihop, otherwise original)
    - Query type classification
    - Reasoning for decomposition
    - Linguistic features
    """
    query = request.query
    logger.info(f"Rewriting query: {query}")

    result = adaptive_rewriter.rewrite(query)

    linguistic_features = result.get("linguistic_features")
    if linguistic_features is not None and hasattr(linguistic_features, "__dict__"):
        # Convert dataclass to dict
        linguistic_features = {
            "query": linguistic_features.query,
            "pos_sequence": linguistic_features.pos_sequence,
            "dep_tree": linguistic_features.dep_tree,
            "entities": linguistic_features.entities,
            "num_tokens": linguistic_features.num_tokens,
            "num_clauses": linguistic_features.num_clauses,
            "syntax_depth": linguistic_features.syntax_depth,
            "has_relative_clauses": linguistic_features.has_relative_clauses,
            "has_conjunctions": linguistic_features.has_conjunctions,
        }

    return RewrittenQuery(
        original_query=result["original"],
        sub_queries=result["queries"],
        is_multihop=result["is_multihop"],
        query_type=QueryType(result.get("query_type", "general")),
        reasoning=result["reasoning"],
        linguistic_features=linguistic_features,
    )
