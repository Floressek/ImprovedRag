"""Pipeline configuration data models."""
from __future__ import annotations

import statistics
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict

from scipy import stats


@dataclass
class PipelineConfig:
    """
    Configuration for a pipeline variant with 4 independent toggles.

    Note: In the new /eval/ablation endpoint, enhanced features (metadata, citations, etc.)
    are controlled by prompt_template:
    - "basic": enhanced features OFF
    - "enhanced": enhanced features ON
    - "auto": enhanced if query_analysis ON, basic if OFF
    - For multihop queries, "multihop" template is always used (enhanced features ON)
    """

    name: str
    description: str
    # Toggle 1: Query Analysis
    query_analysis_enabled: bool = True
    # Toggle 2: Chain of Thought
    cot_enabled: bool = True
    # Toggle 3: Reranking
    reranker_enabled: bool = True
    # Toggle 4: CoVe mode
    cove_mode: str = "off"  # "off", "auto", "metadata", "suggest"
    # Prompt template selection
    prompt_template: str = "auto"  # "basic", "enhanced", "auto" # TODO CHECK IF THIS WORKS WITH MULITHOP
    # LLM provider
    provider: Optional[str] = None  # "api", "ollama", "huggingface", or None (use default)
    # Retrieval parameters
    top_k: Optional[int] = None

    def to_dict(self) -> Dict[str, Union[bool, str, int, None]]:
        """
        Convert to dict for API request.
        Uses aliases compatible with PipelineAblationRequest schema.
        """
        request = {
            "use_query_analysis": self.query_analysis_enabled,
            "use_cot": self.cot_enabled,
            "use_reranker": self.reranker_enabled,
            "cove": self.cove_mode,
            "prompt_template": self.prompt_template,
            "top_k": self.top_k if self.top_k is not None else 15,
        }
        if self.provider:
            request["provider"] = self.provider
        return request


@dataclass
class ConfigResult:
    """Results for a single pipeline configuration."""

    config: PipelineConfig
    evaluation: Any  # BatchEvaluationResult (avoiding circular import)
    run_time_ms: float
    api_responses: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AblationStudyResult:
    """Results from complete ablation study."""

    config_results: List[ConfigResult]
    questions: List[Dict[str, Any]]
    num_questions: int
    total_time_ms: float

    def get_best_config(self, metric: str = "mean_faithfulness") -> ConfigResult:
        """Get configuration with best score for given metric."""
        return max(
            self.config_results,
            key=lambda cr: getattr(cr.evaluation, metric),
        )

    def compare_configs(
            self,
            config_a: str,
            config_b: str,
            metric: str = "mean_faithfulness",
    ) -> Dict[str, Any]:
        """
        Statistical comparison between two configurations.

        Returns t-test results and effect size.

        Note: Only compares questions that were successfully evaluated in BOTH configs.
        """
        # Find configs
        result_a = next((cr for cr in self.config_results if cr.config.name == config_a), None)
        result_b = next((cr for cr in self.config_results if cr.config.name == config_b), None)

        if not result_a or not result_b:
            raise ValueError(f"Configuration not found: {config_a} or {config_b}")

        # Extract per-question scores, filtering out failed questions (score=0 indicates failure)
        metric_name = metric.replace("mean_", "")
        paired_scores_a = []
        paired_scores_b = []

        for r_a, r_b in zip(result_a.evaluation.results, result_b.evaluation.results):
            score_a = getattr(r_a, metric_name)
            score_b = getattr(r_b, metric_name)

            # Only include if both configs have valid scores (non-zero)
            # This ensures paired t-test alignment
            if score_a > 0 and score_b > 0:
                paired_scores_a.append(score_a)
                paired_scores_b.append(score_b)

        if len(paired_scores_a) < 2:
            # Not enough data for t-test
            return {
                "config_a": config_a,
                "config_b": config_b,
                "metric": metric,
                "mean_a": getattr(result_a.evaluation, metric),
                "mean_b": getattr(result_b.evaluation, metric),
                "mean_diff": 0.0,
                "t_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "cohens_d": 0.0,
                "effect_size": "insufficient_data",
                "num_paired_samples": len(paired_scores_a),
            }

        # Get overall means from evaluations (includes all questions)
        mean_a = getattr(result_a.evaluation, metric)
        mean_b = getattr(result_b.evaluation, metric)
        mean_diff = mean_a - mean_b

        # Check if scores are identical (variance = 0 → NaN p-value)
        differences = [a - b for a, b in zip(paired_scores_a, paired_scores_b)]
        if len(set(differences)) == 1:
            # All differences are identical → no variance → p-value = 1.0 (no significant difference)
            t_stat = 0.0
            p_value = 1.0
        else:
            # T-test on paired samples
            t_stat, p_value = stats.ttest_rel(paired_scores_a, paired_scores_b)

        # Effect size (Cohen's d for paired samples)
        # For paired t-test, use standard deviation of differences
        std_diff = statistics.stdev(differences) if len(differences) > 1 else 0.0
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        return {
            "config_a": config_a,
            "config_b": config_b,
            "metric": metric,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_diff": mean_diff,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "cohens_d": cohens_d,
            "effect_size": self._interpret_cohens_d(cohens_d),
            "num_paired_samples": len(paired_scores_a),
        }

    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_questions": self.num_questions,
            "total_time_ms": self.total_time_ms,
            "configs": [
                {
                    "name": cr.config.name,
                    "description": cr.config.description,
                    "config": cr.config.to_dict(),
                    "evaluation": cr.evaluation.to_dict(),
                    "run_time_ms": cr.run_time_ms,
                }
                for cr in self.config_results
            ],
        }


@dataclass
class CheckpointState:
    """State for resuming ablation study from checkpoint."""

    timestamp: str
    completed_configs: List[str]  # Names of completed configs
    config_results: List[Dict[str, Any]]  # Serialized ConfigResult objects
    questions: List[Dict[str, Any]]
    current_config_idx: int
    total_configs: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CheckpointState:
        """Deserialize from dict."""
        return cls(**data)
