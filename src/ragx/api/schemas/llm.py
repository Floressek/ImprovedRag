from __future__ import annotations

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    """Request model for direct LLM generation without RAG."""
    query: str = Field(..., description="Query to generate answer")
    temperature: Optional[float] = Field(
        None,
        description="Sampling temperature",
        ge=0.0,
        le=2.0
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Maximum number of tokens to generate",
        ge=1000,
        le=32000
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Custom instruction to append to prompt"
    )
    chain_of_thought_enabled: Optional[bool] = Field(
        None,
        description="Enable chain of thought"
    )
    template: Optional[str] = Field(
        None,
        description="Prompt template: 'basic' or 'enhanced' (uses pre-built prompts with contexts=[])"
    )
    contexts: Optional[list[Dict[str, Any]]] = Field(
        None,
        description="Optional contexts for template-based prompts (only used if template is specified)"
    )


class LLMResponse(BaseModel):
    """Response model for direct LLM generation."""
    response: str = Field(..., description="Generated text from LLM")
    metadata: Dict[str, Any] = Field(
        ...,
        description="Metadata about the generation"
    )
