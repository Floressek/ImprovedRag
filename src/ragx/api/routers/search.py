from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends

from src.ragx.api.schemas.search import SearchRequest, SearchResult, RerankRequest
from src.ragx.api.dependencies import get_embedder, get_vector_store, get_reranker
from src.ragx.retrieval.embedder.embedder import Embedder
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore
from src.ragx.retrieval.rerankers.reranker import Reranker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("/search", response_model=List[SearchResult])
async def search(
        request: SearchRequest,
        embedder: Embedder = Depends(get_embedder),
        vector_store: QdrantStore = Depends(get_vector_store),
):
    """Search for documents in the vector store."""
    logger.info("Searching for documents in the vector store...")
    query_vector = embedder.embed_query(request.query)

    results = vector_store.search(
        query_vector,
        top_k=request.top_k,
        filter_dict=request.filters,
    )

    output = []
    for doc_id, payload, score in results:
        metadata = payload.get("metadata", {})
        output.append(SearchResult(
            id=str(doc_id),
            doc_title=payload.get("doc_title", "Unknown"),
            text=payload.get("text", ""),
            score=float(score),
            position=payload.get("position", 0),
            url=metadata.get("url"),
            total_chunks=payload.get("total_chunks", 1),
        ))
    logger.info(f"Output: {output}")
    return output


@router.post("/rerank")
async def rerank(
        request: RerankRequest,
        reranker: Reranker = Depends(get_reranker),
        embedder: Embedder = Depends(get_embedder),
        vector_store: QdrantStore = Depends(get_vector_store),
):
    """Rerank documents using the reranker."""
    logger.info("Reranking documents...")

    query_vector = embedder.embed_query(request.query)
    results = vector_store.search(
        query_vector,
        top_k=request.top_k_retrival,
    )

    # tuple to dict conversion should be a separate function
    documents = []
    for doc_id, payload, score in results:
        documents.append({
            "id": str(doc_id),
            "text": payload.get("text", ""),
            "doc_title": payload.get("doc_title", "Unknown"),
            "payload": payload,
            "retrieval_score": float(score),
        })

    reranked_documents = reranker.rerank(
        query=request.query,
        documents=documents,
        top_k=request.top_k_reranker,
    )

    output = []
    for doc, score in reranked_documents:
        payload = doc["payload"]
        metadata = payload.get("metadata", {})
        output.append({
            "id": doc["id"],
            "doc_title": doc["doc_title"],
            "text": doc["text"],
            "retrieval_score": doc.get("retrieval_score"),
            "rerank_score": float(score),
            "position": payload.get("position", 0),
            "total_chunks": payload.get("total_chunks", 1),
            "url": metadata.get("url"),
        })

    return output
