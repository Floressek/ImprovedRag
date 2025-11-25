from src.ragx.ui.constants.types import PipelineConfig

PRESETS = {
    "baseline": PipelineConfig(
        name="Baseline",
        description="🔵 No enhancements (vector search only)",
        query_analysis_enabled=False,
        cot_enabled=False,
        reranker_enabled=False,
        cove_mode="off",
        prompt_template="basic",
        top_k=15
    ),
    "enhanced_full": PipelineConfig(
        name="Enhanced (Full)",
        description="🟢 All enhancements enabled (no CoVe)",
        query_analysis_enabled=True,
        cot_enabled=True,
        reranker_enabled=True,
        cove_mode="auto",
        prompt_template="auto",
        top_k=15
    ),
    "enhanced_cove": PipelineConfig(
        name="Enhanced + CoVe",
        description="🟣 Full pipeline with CoVe auto-correction",
        query_analysis_enabled=True,
        cot_enabled=True,
        reranker_enabled=True,
        cove_mode="auto",
        prompt_template="auto",
        top_k=15
    ),
    "query_only": PipelineConfig(
        name="Query Analysis Only",
        description="🟡 Multihop detection only",
        query_analysis_enabled=True,
        cot_enabled=False,
        reranker_enabled=False,
        cove_mode="off",
        prompt_template="basic",
        top_k=15
    ),
    "reranker_only": PipelineConfig(
        name="Reranker Only",
        description="🟠 Reranking only",
        query_analysis_enabled=False,
        cot_enabled=False,
        reranker_enabled=True,
        cove_mode="off",
        prompt_template="basic",
        top_k=15
    ),
}
