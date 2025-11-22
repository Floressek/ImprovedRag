"""Predefined pipeline configurations for ablation studies."""
from .predefined_configs import (
    BASELINE,
    QUERY_ONLY,
    ENHANCED_ONLY,
    COT_ONLY,
    RERANKER_ONLY,
    MULTIHOP_ONLY,
    COVE_AUTO_ONLY,
    COT_ENHANCED,
    QUERY_RERANK,
    FULL_COVE_AUTO,
    FULL_COVE_METADATA,
    FULL_COVE_SUGGEST,
    FULL_NO_COVE,
    get_all_configs,
)

__all__ = [
    "BASELINE",
    "QUERY_ONLY",
    "ENHANCED_ONLY",
    "COT_ONLY",
    "RERANKER_ONLY",
    "MULTIHOP_ONLY",
    "COVE_AUTO_ONLY",
    "COT_ENHANCED",
    "QUERY_RERANK",
    "FULL_COVE_AUTO",
    "FULL_COVE_METADATA",
    "FULL_COVE_SUGGEST",
    "FULL_NO_COVE",
    "get_all_configs",
]
