from __future__ import annotations

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends

from src.ragx.api.schemas.chat import AskRequest, AskResponse
from src.ragx.api.dependencies import get_baseline_pipeline, get_enhanced_pipeline
from src.ragx.pipelines.baseline import BaselinePipeline
from src.ragx.pipelines.enhanced import EnhancedPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ask", tags=["Chat"])


@router.post("/baseline", response_model=AskResponse)
async def ask_baseline(
        request: AskRequest,
        pipeline: BaselinePipeline = Depends(get_baseline_pipeline),
) -> Dict[str, Any]:
    """Ask q question using the baseline pipeline."""
    chat_history = []
    if request.chat_history:
        chat_history = [msg.model_dump() for msg in request.chat_history]

    result = pipeline.answer(
        query=request.query,
        chat_history=chat_history,
        top_k=request.top_k,
        max_history=request.max_history,
    )

    return result

@router.post("/enhanced", response_model=AskResponse)
async def ask_enhanced(
        request: AskRequest,
        pipeline: EnhancedPipeline = Depends(get_enhanced_pipeline),
) -> Dict[str, Any]:
    """Ask q question using the enhanced pipeline."""
    chat_history = []
    if request.chat_history:
        chat_history = [msg.model_dump() for msg in request.chat_history]

    result = pipeline.answer(
        query=request.query,
        chat_history=chat_history,
        top_k=request.top_k,
        max_history=request.max_history,
    )

    cove_info = None
    if "cove" in result["metadata"]:
        cove_data = result["metadata"]["cove"]
        cove_info = {
            "status": cove_data["status"],
            "needs_correction": cove_data["needs_correction"],
            "num_claims": cove_data.get("num_claims", 0),
            "num_verified": cove_data.get("num_verified", 0),
            "num_refuted": cove_data.get("num_refuted", 0),
            "num_insufficient": cove_data.get("num_insufficient", 0),
            "citations_injected": cove_data.get("citations_injected", 0),
        }

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "metadata": result["metadata"],
        "cove_info": cove_info,
    }

