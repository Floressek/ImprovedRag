from __future__ import annotations

import logging
from functools import lru_cache

from src.ragx.generation.prompts.builder import PromptBuilder
from src.ragx.pipelines.baseline import BaselinePipeline
from src.ragx.pipelines.enhanced import EnhancedPipeline
from src.ragx.pipelines.enhancers.context_fusion import ContextFusionEnhancer
from src.ragx.pipelines.enhancers.reranker import RerankerEnhancer
from src.ragx.retrieval.analyzers.linguistic_analyzer import LinguisticAnalyzer
from src.ragx.retrieval.embedder.embedder import Embedder
from src.ragx.retrieval.rewriters.adaptive_rewriter import AdaptiveQueryRewriter
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore, QdrantConnectionError
from src.ragx.retrieval.rerankers.reranker import Reranker

logger = logging.getLogger(__name__)


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
def get_reranker_enhancer() -> RerankerEnhancer:
    """Get cached reranker enhancer."""
    logger.info("Creating RerankerEnhancer instance")
    return RerankerEnhancer(
        reranker=get_reranker()
    )


@lru_cache(maxsize=1)
def get_baseline_pipeline() -> BaselinePipeline:
    """Get the cached baseline pipeline instance."""
    logger.info("Loading baseline pipeline...")
    return BaselinePipeline(
        embedder=get_embedder(),
        vector_store=get_vector_store()
    )

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
def get_context_fusion() -> ContextFusionEnhancer:
    """Get cached context fusion enhancer."""
    logger.info("Creating ContextFusionEnhancer...")
    return ContextFusionEnhancer()


@lru_cache(maxsize=1)
def get_prompt_builder() -> PromptBuilder:
    """Get cached prompt builder (handles both standard and multihop)."""
    logger.info("Creating PromptBuilder...")
    return PromptBuilder()



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
        context_fusion=get_context_fusion(),
    )
