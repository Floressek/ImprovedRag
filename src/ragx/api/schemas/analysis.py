from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class LinguisticAnalysisRequest(BaseModel):
    """Request model for linguistic analysis."""
    query: str = Field(..., min_length=1, description="Query to analyze")


class LinguisticAnalysisResponse(BaseModel):
    """Response model for linguistic analysis."""
    query: str
    pos_sequence: List[str]
    dep_tree: List[Dict[str, str]]
    entities: List[Dict[str, str]]
    num_tokens: int
    num_clauses: int
    syntax_depth: int
    has_relative_clauses: bool
    has_conjunctions: bool
    analysis_text: str


class MultihopSearchRequest(BaseModel):
    """Request model for multihop search."""
    query: str = Field(..., min_length=1, description="Query to search")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    use_reranker: bool = Field(True, description="Use reranker for multihop search")
    include_linguistic_analysis: bool = Field(
        False,
        description="Include linguistic analysis in results"
    )

class MultihopSearchResult(BaseModel):
    """Single multihop search result."""
    id: str
    doc_title: str
    text: str
    retrieval_score: Optional[float] = None
    local_rerank_score: Optional[float] = None
    fused_score: Optional[float] = None
    global_rerank_score: Optional[float] = None
    final_score: float
    position: int
    url: Optional[str] = None
    fusion_metadata: Optional[Dict[str, Any]] = None


class MultihopSearchResponse(BaseModel):
    """Response model for multihop search."""
    original_query: str
    sub_queries: List[str]
    results: List[MultihopSearchResult]
    linguistic_analysis: Optional[LinguisticAnalysisResponse] = None
    metadata: Dict[str, Any]
