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

    return result

