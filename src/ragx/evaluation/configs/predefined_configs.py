from typing import List

from src.ragx.evaluation.models import PipelineConfig

# === Baseline (all off) ===
BASELINE = PipelineConfig(
    name="baseline",
    description="No enhancements (vector search only)",
    query_analysis_enabled=False,
    cot_enabled=False,
    reranker_enabled=False,
    cove_mode="off",
    prompt_template="basic",
    top_k=10
)

# === Single toggle configs ===
ENHANCED_ONLY = PipelineConfig(
    name="enhanced_only",
    description="Enhanced features only (metadata, quality checks)",
    query_analysis_enabled=False,
    cot_enabled=False,
    reranker_enabled=False,
    cove_mode="off",
    prompt_template="enhanced",
    top_k=15
)

RERANKER_ONLY = PipelineConfig(
    name="reranker_only",
    description="Reranker only",
    query_analysis_enabled=False,
    cot_enabled=False,
    reranker_enabled=True,
    cove_mode="off",
    prompt_template="basic",
    top_k=15
)

# === CoT combinations ===
COT_ENHANCED = PipelineConfig(
    name="cot_enhanced",
    description="CoT + Enhanced Features",
    query_analysis_enabled=False,
    cot_enabled=True,
    reranker_enabled=False,
    cove_mode="off",
    prompt_template="enhanced",
    top_k=15
)

COT_ONLY = PipelineConfig(
    name="cot_only",
    description="Chain of Thought only",
    query_analysis_enabled=False,
    cot_enabled=True,
    reranker_enabled=False,
    cove_mode="off",
    prompt_template="basic",
    top_k=15
)

# === MULTIHOP combinations ===
MULTIHOP_ONLY = PipelineConfig(
    name="multihop_only",
    description="Query Analysis + Reranking",
    query_analysis_enabled=True,
    cot_enabled=False,
    reranker_enabled=True,
    cove_mode="off",
    prompt_template="multihop",
    top_k=15
)

MULTIHOP_COT = PipelineConfig(
    name="multihop+cot",
    description="Multihop detection only",
    query_analysis_enabled=True,
    cot_enabled=True,
    reranker_enabled=True,
    cove_mode="off",
    prompt_template="multihop",
    top_k=15
)



# === CoVe mode variations ===
COVE_AUTO_ONLY = PipelineConfig(
    name="cove_auto_only",
    description="CoVe auto-correction only",
    query_analysis_enabled=False,
    cot_enabled=True,
    reranker_enabled=False,
    cove_mode="auto",
    prompt_template="basic",
    top_k=15
)

FULL_COVE_AUTO = PipelineConfig(
    name="full_cove_auto",
    description="Full pipeline with CoVe auto-correction",
    query_analysis_enabled=True,
    cot_enabled=True,
    reranker_enabled=True,
    cove_mode="auto",
    prompt_template="multihop",
    top_k=15
)

FULL_COVE_METADATA = PipelineConfig(
    name="full_cove_metadata",
    description="Full pipeline with CoVe metadata-only",
    query_analysis_enabled=True,
    cot_enabled=True,
    reranker_enabled=True,
    cove_mode="metadata",
    prompt_template="multihop",
    top_k=15
)

FULL_COVE_SUGGEST = PipelineConfig(
    name="full_cove_suggest",
    description="Full pipeline with CoVe suggest mode",
    query_analysis_enabled=True,
    cot_enabled=True,
    reranker_enabled=True,
    cove_mode="suggest",
    prompt_template="multihop",
    top_k=15
)

# === Full (no CoVe) ===
FULL_NO_COVE = PipelineConfig(
    name="full_no_cove",
    description="Full pipeline without CoVe",
    query_analysis_enabled=True,
    cot_enabled=True,
    reranker_enabled=True,
    cove_mode="off",
    prompt_template="multihop",
    top_k=15
)


def get_all_configs() -> List[PipelineConfig]:
    """Get all predefined configurations (12 configs total)."""
    return [
        BASELINE,
        ENHANCED_ONLY,
        COT_ONLY,
        RERANKER_ONLY,
        COVE_AUTO_ONLY,
        COT_ENHANCED,
        MULTIHOP_ONLY,
        MULTIHOP_COT,
        FULL_NO_COVE,
        FULL_COVE_AUTO,
        FULL_COVE_METADATA,
        FULL_COVE_SUGGEST,
    ]
