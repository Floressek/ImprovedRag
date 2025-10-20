from __future__ import annotations

import logging
from typing import Dict, Any

from fastapi import APIRouter

from src.ragx.api.schemas import LLMRequest, LLMResponse
from src.ragx.generation.inference import LLMInference

logger = logging.getLogger(__name__)

router = APIRouter(tags=["LLM"])


@router.post("/generate", response_model=LLMResponse)
async def generate_llm(
        request: LLMRequest,
) -> Dict[str, Any]:
    """
    Generate text using LLM without RAG pipeline - direct conversation with LLM.

    Args:
        request: LLMRequest with prompt, optional temperature, max_tokens, and custom_instructions

    Returns:
        Generated text response
    """
    logger.info(f"📝 LLM generate request: prompt_length={len(request.query)}")
    final_prompt = request.query
    if request.system_prompt:
        final_prompt = f"{request.system_prompt}\n\n{final_prompt}"

    llm = LLMInference()

    response = llm.generate(
        prompt=final_prompt,
        temperature=request.temperature,
        max_new_tokens=request.max_tokens,
        chain_of_thought_enabled=request.chain_of_thought_enabled,
    )

    logger.info(f"✅ LLM generated: response_length={len(response)}")

    used_temperature = request.temperature if request.temperature is not None else llm.temperature
    used_max_tokens = request.max_tokens if request.max_tokens is not None else llm.max_new_tokens

    return {
        "response": response,
        "metadata": {
            "query": len(request.query),
            "response_length": len(response),
            "temperature": used_temperature,
            "max_tokens": used_max_tokens,
            "provider": llm.provider,
        }
    }
