from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from src.ragx.api.schemas.health import HealthResponse
from src.ragx.api.dependencies import get_embedder, get_vector_store, get_reranker, get_llm, get_linguistic_analyzer, \
    get_adaptive_rewriter, get_prompt_builder, get_multihop_reranker, get_cove_enhancer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/info", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
        embedder=Depends(get_embedder),
        vector_store=Depends(get_vector_store),
        reranker=Depends(get_reranker),
        llm=Depends(get_llm),
        ling_analyzer=Depends(get_linguistic_analyzer),
        adapt_rewriter=Depends(get_adaptive_rewriter),
        prompt_builder=Depends(get_prompt_builder),
        multihop_reranker=Depends(get_multihop_reranker),
        cove_enhancer=Depends(get_cove_enhancer),
):
    """Health check endpoint."""

    elements_status = {
        "embedder": embedder is not None,
        "reranker": reranker is not None,
        "llm": llm is not None,
        "ling_analyzer": ling_analyzer is not None,
        "adapt_rewriter": adapt_rewriter is not None,
        "prompt_builder": prompt_builder is not None,
        "multihop_reranker": multihop_reranker is not None,
        "cove_enhancer": cove_enhancer is not None,
    }

    collection_info = vector_store.get_collection_info()

    return HealthResponse(
        status="ok",
        models=elements_status,
        collection={
            "name": collection_info["name"],
            "points_count": collection_info["points_count"],
            "vector_size": collection_info["vector_size"],
        },
    )
