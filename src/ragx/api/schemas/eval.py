from __future__ import annotations

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class PipelineAblationRequest(BaseModel):
    """Request for pipeline ablation study."""
    query: str = Field(..., description="Query to process")

    # Toggles
    use_query_analysis: bool = Field(
        True,
        description="Enable query analysis and multihop decomposition"
    )
    use_reranker: bool = Field(
        True,
        description="Enable reranking (multihop or standard)"
    )
    use_cove: bool = Field(
        False,
        description="Enable CoVe verification"
    )
    use_cot: bool = Field(
        True,
        description="Enable Chain-of-Thought for Ollama models"
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


class PipelineAblationResponse(BaseModel):
    """Response from pipeline ablation."""
    answer: str
    sources: list[Dict[str, Any]]
    metadata: Dict[str, Any] = Field(
        ...,
        description="Detailed metadata about each stage and what was enabled/disabled"
    )