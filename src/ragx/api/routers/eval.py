from __future__ import annotations

import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, Depends

from src.ragx.api.schemas.eval import PipelineAblationRequest, PipelineAblationResponse
from src.ragx.api.dependencies import (
    get_embedder,
    get_vector_store,
    get_linguistic_analyzer,
    get_adaptive_rewriter,
    get_reranker_enhancer,
    get_multihop_reranker,
    get_cove_enhancer,
)
from src.ragx.generation.inference import LLMInference
from src.ragx.generation.prompts.builder import PromptBuilder, PromptConfig
from src.ragx.generation.prompts.utils.citation_remapper import remap_citations
from src.ragx.retrieval.analyzers.linguistic_analyzer import LinguisticAnalyzer
from src.ragx.retrieval.embedder.embedder import Embedder
from src.ragx.retrieval.rewriters.adaptive_rewriter import AdaptiveQueryRewriter
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore
from src.ragx.pipelines.enhancers.reranker import RerankerEnhancer
from src.ragx.pipelines.enhancers.multihop_reranker import MultihopRerankerEnhancer
from src.ragx.pipelines.enhancers.cove import CoVeEnhancer
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["Evaluation"])
@router.post("/ablation", response_model=PipelineAblationResponse)
async def pipeline_ablation(
        request: PipelineAblationRequest,
        embedder: Embedder = Depends(get_embedder),
        vector_store: QdrantStore = Depends(get_vector_store),
        linguistic_analyzer: LinguisticAnalyzer = Depends(get_linguistic_analyzer),
        adaptive_rewriter: AdaptiveQueryRewriter = Depends(get_adaptive_rewriter),
        reranker_enhancer: RerankerEnhancer = Depends(get_reranker_enhancer),
        multihop_reranker: MultihopRerankerEnhancer = Depends(get_multihop_reranker),
        cove_enhancer: CoVeEnhancer = Depends(get_cove_enhancer),
) -> Dict[str, Any]:
    """
    Pipeline ablation study - toggle each stage on/off.
    Allows testing different combinations of:
    - Query analysis & multihop decomposition
    - Reranking (standard or multihop)
    - Prompt templates (basic/enhanced/multihop/auto)
    - Chain-of-Thought
    - CoVe verification
    - LLM provider
    Args:
        request: Configuration for which stages to enable/disable
    Returns:
        Answer with detailed metadata about what was enabled/disabled
    """
    start = time.time()
    query = request.query

    logger.info(
        f"Ablation request: query_analysis={request.use_query_analysis}, "
        f"reranker={request.use_reranker}, cove={request.use_cove}, "
        f"template={request.prompt_template}, cot={request.use_cot}"
    )

    # metadata = {
    #     "ablation_config": {
    #         "use_query_analysis": request.use_query_analysis,
    #         "use_reranker": request.use_reranker,
    #         "use_cove": request.use_cove,
    #         "use_cot": request.use_cot,
    #         "prompt_template": request.prompt_template,
    #         "provider": request.provider if request.provider else settings.llm.provider,
    #     }
    # }


