from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class AskRequest(BaseModel):
    """Ask request model. /ask endpoint."""
    query: str = Field(..., description="Question to ask", min_length=1)
    chat_history: Optional[List[ChatMessage]] = Field(
        None,
        description="Optional chat history for context"
    )
    max_history: Optional[int] = Field(
        5,
        description="Maximum number of chat messages to include in context",
        ge=0, le=20
    )
    top_k: Optional[int] = Field(None, description="Override top_k setting", ge=1, le=50)


class SourceInfo(BaseModel):
    """Source document info."""
    id: str
    doc_title: str
    text: str
    position: int
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    url: Optional[str] = None # TO BE ADDED


class AskResponse(BaseModel):
    """Response model for /ask endpoints."""
    answer: str
    sources: List[SourceInfo]
    metadata: Dict[str, Any]


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
    # documents: List[Dict[str, Any]]
    top_k_retrival: int = Field(5, ge=1, le=100)
    top_k_reranker: int = Field(5, ge=1, le=50)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models: Dict[str, bool]
    collection: Dict[str, Any]
