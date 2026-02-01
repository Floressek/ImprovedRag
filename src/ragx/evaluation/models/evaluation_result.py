from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    """Results from RAGAS evaluation with custom metrics."""

    # RAGAS metrics
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    # Custom metrics
    latency_ms: float
    sources_count: int  # Final sources in response (after reranking/CoVe)
    num_candidates: float  # Retrieved candidates (avg per sub-query for multihop, total for single)
    multihop_coverage: float  # Ratio of sub-queries with at least 1 retrieved doc

    # Additional stats
    num_contexts: int
    query_type: Optional[str] = None
    is_multihop: bool = False
    num_sub_queries: int = 1

    # Per-question details (optional)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ragas_faithfulness": self.faithfulness,
            "ragas_answer_relevancy": self.answer_relevancy,
            "ragas_context_precision": self.context_precision,
            "ragas_context_recall": self.context_recall,
            "custom_latency_ms": self.latency_ms,
            "custom_sources_count": self.sources_count,
            "custom_num_candidates": self.num_candidates,
            "custom_multihop_coverage": self.multihop_coverage,
            "num_contexts": self.num_contexts,
            "query_type": self.query_type,
            "is_multihop": self.is_multihop,
            "num_sub_queries": self.num_sub_queries,
        }


@dataclass
class BatchEvaluationResult:
    """Aggregated results from evaluating multiple questions."""

    # Aggregated RAGAS metrics (mean)
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    mean_context_recall: float

    # Aggregated custom metrics (mean)
    mean_latency_ms: float
    mean_sources_count: float
    mean_num_candidates: float
    mean_multihop_coverage: float

    # Confidence intervals (95% CI) - (lower, upper)
    ci_faithfulness: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    ci_answer_relevancy: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    ci_context_precision: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    ci_context_recall: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    # Standard deviations
    std_faithfulness: float = 0.0
    std_answer_relevancy: float = 0.0
    std_context_precision: float = 0.0
    std_context_recall: float = 0.0

    # Per-question results
    results: List[EvaluationResult] = field(default_factory=list)

    # Statistics
    num_questions: int = 0
    num_multihop: int = 0
    num_simple: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_faithfulness": self.mean_faithfulness,
            "mean_answer_relevancy": self.mean_answer_relevancy,
            "mean_context_precision": self.mean_context_precision,
            "mean_context_recall": self.mean_context_recall,
            "mean_latency_ms": self.mean_latency_ms,
            "mean_sources_count": self.mean_sources_count,
            "mean_num_candidates": self.mean_num_candidates,
            "mean_multihop_coverage": self.mean_multihop_coverage,
            "ci_faithfulness": self.ci_faithfulness,
            "ci_answer_relevancy": self.ci_answer_relevancy,
            "ci_context_precision": self.ci_context_precision,
            "ci_context_recall": self.ci_context_recall,
            "std_faithfulness": self.std_faithfulness,
            "std_answer_relevancy": self.std_answer_relevancy,
            "std_context_precision": self.std_context_precision,
            "std_context_recall": self.std_context_recall,
            "num_questions": self.num_questions,
            "num_multihop": self.num_multihop,
            "num_simple": self.num_simple,
        }
