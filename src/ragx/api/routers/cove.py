from __future__ import annotations

import logging
from typing import Dict, Any, Coroutine

from fastapi import APIRouter, Depends

from src.ragx.api.schemas.cove import CoVeVerificationRequest, CoVeVerificationResponse, ClaimInfo
from src.ragx.api.dependencies import get_cove_enhancer
from src.ragx.pipelines.enhancers.cove import CoVeEnhancer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cove", tags=["CoVe"])


@router.post("/verify", response_model=CoVeVerificationResponse)
async def verify_answer(
        request: CoVeVerificationRequest,
        cove_enhancer: CoVeEnhancer = Depends(get_cove_enhancer),
) -> CoVeVerificationResponse:
    """
    Verify an answer using CoVe (Chain-of-Verification).

    This endpoint allows you to test CoVe verification independently
    from the full RAG pipeline.

    Args:
        request: Query, answer, and contexts to verify against

    Returns:
        CoVe verification result with claims, status, and optional correction
    """
    logger.info(f"CoVe verification request: query={request.query[:50]}, answer_len={len(request.answer)}")

    result = cove_enhancer.verify(
        query=request.query,
        answer=request.answer,
        contexts=request.contexts,
    )

    claims_formatted = [
        ClaimInfo(
            text=v.claim.text,
            label=v.label,
            confidence=v.confidence,
            reasoning=v.reasoning,
            has_citations=v.claim.has_citations,
        )
        for v in result.verifications
    ]

    logger.info(f"CoVe verification complete: status={result.status}, num_claims={len(claims_formatted)}")

    return CoVeVerificationResponse(
        original_answer=result.original_answer,
        corrected_answer=result.corrected_answer,
        status=str(result.status),
        needs_correction=result.needs_correction,
        claims=claims_formatted,
        metadata=result.metadata,
    )
