from __future__ import annotations

import logging
from functools import lru_cache

from src.ragx.pipelines.baseline import BaselinePipeline
from src.ragx.pipelines.enhanced import EnhancedPipeline
from src.ragx.pipelines.enhancers.reranker import RerankerEnhancer
from src.ragx.retrieval.embedder.embedder import Embedder
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore
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
    return QdrantStore()


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
def get_enhanced_pipeline() -> EnhancedPipeline:
    """Get the cached enhanced pipeline instance."""
    logger.info("Loading enhanced pipeline...")
    return EnhancedPipeline(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        reranker_enhancer=RerankerEnhancer(reranker=get_reranker_enhancer())
    )
