from __future__ import annotations

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class PipelineAblationRequest(BaseModel):
    """Request for pipeline ablation study with 4 independent toggles."""
    query: str = Field(..., description="Query to process")

    # Toggle 1: Query Analysis - multihop detection + adaptive rewriting
    query_analysis_enabled: bool = Field(
        True,
        alias="use_query_analysis",
        description="Toggle 1: Enable query analysis (OFF = always single query)"
    )

    # Toggle 2: Chain of Thought
    cot_enabled: bool = Field(
        True,
        alias="use_cot",
        description="Toggle 2: Enable Chain-of-Thought reasoning"
    )

    # Toggle 3: Reranking - 3-stage (multihop) or standard (single)
    reranker_enabled: bool = Field(
        True,
        alias="use_reranker",
        description="Toggle 3: Enable reranking (type auto-selected: multihop → 3-stage, single → standard)"
    )

    # Toggle 4: CoVe mode - "off", "auto", "metadata", "suggest"
    cove_mode: str = Field(
        "off",
        alias="cove",
        description="Toggle 4: CoVe mode ('off', 'auto', 'metadata', 'suggest')"
    )

    # Prompt template selection
    prompt_template: str = Field(
        "auto",
        description="Prompt template: 'basic', 'enhanced', 'auto' (auto = enhanced if query_analysis ON, else basic; multihop always uses multihop template)"
    )

    # Provider
    provider: Optional[str] = Field(
        None,
        description="LLM provider: 'api', 'ollama', 'huggingface', or None (use default from settings)"
    )

    # Retrieval params
    top_k: int = Field(
        10,
        description="Number of final contexts to use",
        ge=1,
        le=50
    )

    class Config:
        populate_by_name = True


class PipelineAblationResponse(BaseModel):
    """Response from pipeline ablation."""
    answer: str
    sources: list[Dict[str, Any]] = Field(
        default_factory=list,
        description="Unique sources with citations"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Metrics and config details (total_time_ms, is_multihop, query_type, template_used, etc.)"
    )