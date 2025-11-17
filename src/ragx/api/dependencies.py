from __future__ import annotations

import logging
from functools import lru_cache

from src.ragx.generation.inference import LLMInference
from src.ragx.generation.prompts.builder import PromptBuilder
from src.ragx.pipelines.baseline import BaselinePipeline
from src.ragx.pipelines.enhanced import EnhancedPipeline
from src.ragx.pipelines.enhancers.cove import CoVeEnhancer
from src.ragx.pipelines.enhancers.multihop_reranker import MultihopRerankerEnhancer
from src.ragx.pipelines.enhancers.reranker import RerankerEnhancer
from src.ragx.retrieval.analyzers.linguistic_analyzer import LinguisticAnalyzer
from src.ragx.retrieval.embedder.embedder import Embedder
from src.ragx.retrieval.rewriters.adaptive_rewriter import AdaptiveQueryRewriter
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore, QdrantConnectionError
from src.ragx.retrieval.rerankers.reranker import Reranker
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Core Components (Cached)
# ============================================================================


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Get the cached embedder instance."""
    logger.info("Loading embedder...")
    return Embedder()


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantStore:
    """Get the cached vector store instance."""
    logger.info("Loading vector store...")
    try:
        return QdrantStore()
    except QdrantConnectionError:
        raise  # handled by main lifespan
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Get the cached reranker instance."""
    logger.info("Loading reranker...")
    return Reranker()


@lru_cache(maxsize=1)
def get_llm() -> LLMInference:
    """Get the cached LLM inference instance."""
    logger.info("Loading LLM...")
    return LLMInference()

# ============================================================================
# Query Processing (Cached)
# ============================================================================


@lru_cache(maxsize=1)
def get_linguistic_analyzer() -> LinguisticAnalyzer:
    """Get cached linguistic analyzer."""
    logger.info("Loading linguistic analyzer...")
    return LinguisticAnalyzer()


@lru_cache(maxsize=1)
def get_adaptive_rewriter() -> AdaptiveQueryRewriter:
    """Get cached adaptive rewriter."""
    logger.info("Loading adaptive rewriter...")
    return AdaptiveQueryRewriter(
        analyzer=get_linguistic_analyzer()
    )


@lru_cache(maxsize=1)
def get_prompt_builder() -> PromptBuilder:
    """Get cached prompt builder (handles both standard and multihop)."""
    logger.info("Creating PromptBuilder...")
    return PromptBuilder()


# ============================================================================
# Enhancers (Cached)
# ============================================================================



@lru_cache(maxsize=1)
def get_reranker_enhancer() -> RerankerEnhancer:
    """Get cached reranker enhancer."""
    logger.info("Creating RerankerEnhancer instance")
    return RerankerEnhancer(
        reranker=get_reranker()
    )

@lru_cache(maxsize=1)
def get_multihop_reranker() -> MultihopRerankerEnhancer:
    """Get cached multihop reranker."""
    logger.info("Creating MultihopRerankerEnhancer...")
    reranker = get_reranker()
    return MultihopRerankerEnhancer(
        reranker=reranker,
        top_k_per_subquery=settings.multihop.top_k_per_subquery,
        final_top_k=settings.retrieval.context_top_n,
        fusion_strategy=settings.multihop.fusion_strategy,
        global_rerank_weight=settings.multihop.global_rerank_weight,
    )

@lru_cache(maxsize=1)
def get_cove_enhancer() -> CoVeEnhancer:
    """Get cached CoVe enhancer."""
    logger.info("Creating CoVeEnhancer...")
    return CoVeEnhancer(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        reranker=get_reranker(),
    )

# ============================================================================
# Pipelines
# ============================================================================

@lru_cache(maxsize=1)
def get_baseline_pipeline() -> BaselinePipeline:
    """Get the cached baseline pipeline instance."""
    logger.info("Loading baseline pipeline...")
    return BaselinePipeline(
        embedder=get_embedder(),
        vector_store=get_vector_store()
    )

@lru_cache(maxsize=1)
def get_enhanced_pipeline() -> EnhancedPipeline:
    """Get the cached enhanced pipeline instance."""
    logger.info("Loading enhanced pipeline...")
    return EnhancedPipeline(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        linguistic_analyzer=get_linguistic_analyzer(),
        adaptive_rewriter=get_adaptive_rewriter(),
        reranker_enhancer=get_reranker_enhancer(),
        multihop_reranker=get_multihop_reranker(),
        cove_enhancer=get_cove_enhancer(),
    )
