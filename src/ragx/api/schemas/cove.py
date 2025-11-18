from __future__ import annotations

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class CoVeVerificationRequest(BaseModel):
    """Request for solo CoVe verification."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Answer to verify")
    contexts: List[Dict[str, Any]] = Field(
        ...,
        description="Retrieved contexts (sources) to verify against"
    )


class ClaimInfo(BaseModel):
    """Information about a single claim."""
    text: str
    label: str  # supports, refutes, insufficient
    confidence: float
    reasoning: str
    has_citations: bool


class CoVeVerificationResponse(BaseModel):
    """Response from CoVe verification."""
    original_answer: str
    corrected_answer: Optional[str]
    status: str
    needs_correction: bool
    claims: List[ClaimInfo]
    metadata: Dict[str, Any]