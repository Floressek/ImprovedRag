from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends

from src.ragx.api.schemas import SearchRequest, SearchResult, RerankRequest
from src.ragx.api.dependencies import get_embedder, get_vector_store, get_reranker
from src.ragx.retrieval.embedder.embedder import Embedder
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore
from src.ragx.retrieval.rerankers.reranker import Reranker

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Search"])


@router.post("/search", response_model=List[SearchResult])
async def search(
        request: SearchRequest,
        embedder: Embedder = Depends(get_embedder),
        vector_store: QdrantStore = Depends(get_vector_store),
):
    """Search for documents in the vector store."""
    logger.info(f"Searching for documents in the vector store...")
