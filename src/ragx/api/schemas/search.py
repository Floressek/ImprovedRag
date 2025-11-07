from __future__ import annotations

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request model for /search endpoint."""
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    """Single search result."""
    id: str
    doc_title: str
    text: str
    score: float
    position: int
    url: Optional[str] = None
    total_chunks: Optional[int] = None


class RerankRequest(BaseModel):
    """Request model for /rerank endpoint."""
    query: str = Field(..., min_length=1)
    top_k_retrival: int = Field(5, ge=1, le=300)
    top_k_reranker: int = Field(5, ge=1, le=50)