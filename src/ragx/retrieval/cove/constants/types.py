from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.ragx.retrieval.cove.constants.cove_status import CoVeStatus


@dataclass
class Claim:
    """Single factual extracted claim."""
    text: str
    claim_id: int
    original_sentence: Optional[str] = None
    has_citations: bool = False
    citations: List[int] = field(default_factory=list)

@dataclass
class Evidence:
    """Single evidence document."""
    text: str
    doc_id: str
    score: float
    doc_title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Verification:
    """Single verification document."""
    claim: Claim
    label: str # "supports", "refutes", "insufficient"
    confidence: float
    reasoning: str
    evidences: List[Evidence] = field(default_factory=list)

@dataclass
class CoVeResult:
    """Complete CoVe verification result."""
    original_answer: str
    corrected_answer: Optional[str]
    verifications: List[Verification]
    status: CoVeStatus
    needs_correction: bool
    metadata: Dict[str, Any] = field(default_factory=dict)