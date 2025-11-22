"""Helper functions for RAGx Chat UI."""

import requests
import streamlit as st
from typing import Dict, Any, List

from .types import PipelineConfig, PipelineStep, StepTiming


def estimate_step_timings(config: PipelineConfig) -> StepTiming:
    """
    Estimate realistic duration for each pipeline step.

    Based on real Enhanced pipeline performance (~30 seconds total).
    """
    timings = StepTiming()

    # Query Analysis: ~1-2s
    if config.query_analysis_enabled:
        timings.query_analysis = 1.5

    # Retrieval: ~5-8s (vector search + embeddings)
    timings.retrieval = 6.0

    # Reranking: ~3-5s (especially for multihop 3-stage)
    if config.reranker_enabled:
        timings.reranking = 4.0

    # Generation: ~10-15s (longest step, LLM inference)
    # CoT makes it even longer
    if config.cot_enabled:
        timings.generation = 14.0
    else:
        timings.generation = 8.0

    # CoVe: ~5-8s (additional verification queries)
    if config.cove_mode != "off":
        timings.cove = 6.5

    return timings


def get_pipeline_steps(config: PipelineConfig) -> List[PipelineStep]:
    """
    Get list of pipeline steps with their messages and timing estimates.

    Returns only enabled steps with proper numbering.
    """
    timings = estimate_step_timings(config)

    all_steps = [
        PipelineStep(
            key="query_analysis",
            message="🔍 Analyzing query",
            enabled=config.query_analysis_enabled,
            estimated_duration=timings.query_analysis
        ),
        PipelineStep(
            key="retrieval",
            message="📥 Retrieving candidates",
            enabled=True,
            estimated_duration=timings.retrieval
        ),
        PipelineStep(
            key="reranking",
            message="📊 Reranking results",
            enabled=config.reranker_enabled,
            estimated_duration=timings.reranking
        ),
        PipelineStep(
            key="generation",
            message="💭 Generating answer",
            enabled=True,
            estimated_duration=timings.generation
        ),
        PipelineStep(
            key="cove",
            message="✅ Verifying with CoVe",
            enabled=config.cove_mode != "off",
            estimated_duration=timings.cove
        ),
    ]

    # Number only enabled steps
    enabled_count = sum(1 for step in all_steps if step.enabled)
    step_num = 1

    result = []
    for step in all_steps:
        if step.enabled:
            numbered_msg = f"**Step {step_num}/{enabled_count}:** {step.message}..."
            result.append(PipelineStep(
                key=step.key,
                message=numbered_msg,
                enabled=True,
                estimated_duration=step.estimated_duration
            ))
            step_num += 1
        else:
            result.append(PipelineStep(
                key=step.key,
                message=f"{step.message} (skipped)",
                enabled=False,
                estimated_duration=0.0
            ))

    return result


def call_rag_api(query: str, config: PipelineConfig, api_url: str) -> Dict[str, Any]:
    """
    Call RAG API with given config.

    Raises:
        requests.exceptions.RequestException: On API errors
        TimeoutError: If request times out
    """
    request_data = {
        "query": query,
        "use_query_analysis": config.query_analysis_enabled,
        "use_cot": config.cot_enabled,
        "use_reranker": config.reranker_enabled,
        "cove": config.cove_mode,
        "prompt_template": config.prompt_template,
        "top_k": config.top_k,
    }

    response = requests.post(
        f"{api_url}/eval/ablation",
        json=request_data,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def update_session_stats(config_name: str, total_time_ms: float):
    """Update session statistics with query results."""
    stats = st.session_state.session_stats
    stats["total_queries"] += 1
    stats["total_time_ms"] += total_time_ms

    if config_name not in stats["configs_used"]:
        stats["configs_used"][config_name] = 0
    stats["configs_used"][config_name] += 1
