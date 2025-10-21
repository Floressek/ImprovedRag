from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from src.ragx.api.schemas import HealthResponse
from src.ragx.api.dependencies import get_embedder, get_vector_store, get_reranker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/info", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
        embedder=Depends(get_embedder),
        vector_store=Depends(get_vector_store),
        reranker=Depends(get_reranker),
):
    """Health check endpoint."""

    models_status = {
        "embedder": embedder is not None,
        "reranker": reranker is not None,
    }

    collection_info = vector_store.get_collection_info()

    return HealthResponse(
        status="ok",
        models=models_status,
        collection={
            "name": collection_info["name"],
            "points_count": collection_info["points_count"],
            "vector_size": collection_info["vector_size"],
        },
    )
