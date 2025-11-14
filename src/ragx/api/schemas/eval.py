from __future__ import annotations

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class PipelineAblationRequest(BaseModel):
    """Request for pipeline ablation study."""
    query: str = Field(..., description="Query to process")

    # Toggles (aligned with ablation_study.py)
    query_analysis_enabled: bool = Field(
        True,
        alias="use_query_analysis",
        description="Enable query analysis and multihop decomposition"
    )
    reranker_enabled: bool = Field(
        True,
        alias="use_reranker",
        description="Enable reranking (multihop or standard)"
    )
    cove_enabled: bool = Field(
        False,
        alias="use_cove",
        description="Enable CoVe verification"
    )
    multihop_enabled: bool = Field(
        True,
        alias="use_multihop",
        description="Enable multihop decomposition"
    )
    use_cot: bool = Field(
        True,
        description="Enable Chain-of-Thought for generation"
    )

    # Prompt engineering
    prompt_template: str = Field(
        "auto",
        description="Prompt template: 'basic', 'enhanced', 'multihop', 'auto' (auto-select based on query analysis)"
    )

    # Provider
    provider: Optional[str] = Field(
        None,
        description="LLM provider: 'api', 'ollama', 'huggingface', or None (use default from settings)"
    )

    # Retrieval params
    top_k: int = Field(
        8,
        description="Number of final contexts to use",
        ge=1,
        le=50
    )

    class Config:
        populate_by_name = True  # Allow both field name and alias


class PipelineAblationResponse(BaseModel):
    """Response from pipeline ablation."""
    answer: str
    contexts: list[str] = Field(
        ...,
        description="Retrieved context chunks (text only)"
    )
    context_details: list[Dict[str, Any]] = Field(
        ...,
        description="Full context details with URLs, scores, etc."
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Sub-queries from multihop decomposition"
    )
    sources: list[Dict[str, Any]] = Field(
        default_factory=list,
        description="Unique sources with citations"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Metrics and config details (total_time_ms, is_multihop, query_type, etc.)"
    )