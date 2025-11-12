from __future__ import annotations

import logging
from typing import Dict, Any

from fastapi import APIRouter

from src.ragx.api.schemas.llm import LLMRequest, LLMResponse
from src.ragx.generation.inference import LLMInference
from src.ragx.generation.prompts.builder import PromptBuilder, PromptConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM"])


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

    # If template is specified, use PromptBuilder
    if request.template:
        logger.info(f"Using template: {request.template}")
        prompt_builder = PromptBuilder()

        # Use provided contexts or empty list
        contexts = request.contexts if request.contexts else []

        # Build prompt config
        config = PromptConfig(
            use_cot=request.chain_of_thought_enabled if request.chain_of_thought_enabled is not None else True,
            include_metadata=True,
            strict_citations=True,
            detect_language=True,
        )

        # Build prompt based on template
        if request.template == "basic":
            final_prompt = prompt_builder.build(
                query=request.query,
                contexts=contexts,
                template_name="basic",
                config=config,
            )
        elif request.template == "enhanced":
            final_prompt = prompt_builder.build(
                query=request.query,
                contexts=contexts,
                template_name="enhanced",
                config=config,
            )
        else:
            # Fallback to basic
            final_prompt = prompt_builder.build(
                query=request.query,
                contexts=contexts,
                template_name="basic",
                config=config,
            )
    else:
        # Original behavior - direct prompt
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
            "template": request.template if request.template else "none",
            "num_contexts": len(request.contexts) if request.contexts else 0,
        }
    }
