from __future__ import annotations

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class PipelineAblationRequest(BaseModel):
    """Request for pipeline ablation study with 5 independent toggles."""
    query: str = Field(..., description="Query to process")

    # 5 Independent Toggles = 32 base permutations
    # Toggle 1: Query Analysis - multihop detection + template choice
    query_analysis_enabled: bool = Field(
        True,
        alias="use_query_analysis",
        description="Toggle 1: Enable query analysis (OFF = always single query, template 'enhanced')"
    )

    # Toggle 2: Enhanced Features - metadata, quality checks
    enhanced_features_enabled: bool = Field(
        True,
        alias="use_enhanced_features",
        description="Toggle 2: Enable enhanced features (metadata, quality checks, contradictions)"
    )

    # Toggle 3: Chain of Thought
    cot_enabled: bool = Field(
        True,
        alias="use_cot",
        description="Toggle 3: Enable Chain-of-Thought reasoning"
    )

    # Toggle 4: Reranking - 3-stage (multihop) or standard (single)
    reranker_enabled: bool = Field(
        True,
        alias="use_reranker",
        description="Toggle 4: Enable reranking (type auto-selected: multihop → 3-stage, single → standard)"
    )

    # Toggle 5: CoVe mode - "off", "auto", "metadata", "suggest"
    cove_mode: str = Field(
        "off",
        alias="cove",
        description="Toggle 5: CoVe mode ('off', 'auto', 'metadata', 'suggest')"
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
    # contexts: list[str] = Field(
    #     ...,
    #     description="Retrieved context chunks (text only)"
    # )
    # context_details: list[Dict[str, Any]] = Field(
    #     ...,
    #     description="Full context details with URLs, scores, etc."
    # )
    # sub_queries: list[str] = Field(
    #     default_factory=list,
    #     description="Sub-queries from multihop decomposition"
    # )
    sources: list[Dict[str, Any]] = Field(
        default_factory=list,
        description="Unique sources with citations"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Metrics and config details (total_time_ms, is_multihop, query_type, etc.)"
    )