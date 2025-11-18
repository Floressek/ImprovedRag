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
    top_k: Optional[int] = Field(
        None,
        description="Override top_k setting",
        ge=1,
        le=50
    )


class SourceInfo(BaseModel):
    """Source document info."""
    id: str
    doc_title: str
    text: str
    position: int
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    local_rerank_score: Optional[float] = None
    fused_score: Optional[float] = None
    global_rerank_score: Optional[float] = None
    final_score: Optional[float] = None
    url: Optional[str] = None
    fusion_metadata: Optional[Dict[str, Any]] = None


class AskResponse(BaseModel):
    """Response model for /ask endpoints."""
    answer: str
    sources: List[SourceInfo]
    metadata: Dict[str, Any]
    cove_info: Optional[CoVeInfo] = None

class CoVeInfo(BaseModel):
    """CoVe verification info."""
    status: str
    needs_correction: bool
    num_claims: int
    num_verified: int
    num_refuted: int
    num_insufficient: int
    citations_injected: bool