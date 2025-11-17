"""
Ablation study runner for RAG pipeline evaluation.

Tests different pipeline configurations systematically:
- Baseline (no enhancements)
- +Query Analysis
- +Reranker
- +CoVe
- Full pipeline (all enhancements)
"""

from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import statistics

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy import stats

from src.ragx.evaluation.ragas_evaluator import RAGASEvaluator, BatchEvaluationResult
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a pipeline variant with 5 independent toggles."""

    name: str
    description: str
    # Toggle 1: Query Analysis
    query_analysis_enabled: bool = True
    # Toggle 2: Enhanced Features
    enhanced_features_enabled: bool = True
    # Toggle 3: Chain of Thought
    cot_enabled: bool = True
    # Toggle 4: Reranking
    reranker_enabled: bool = True
    # Toggle 5: CoVe mode
    cove_mode: str = "off"  # "off", "auto", "metadata", "suggest"

    def to_dict(self) -> Dict[str, Union[bool, str]]:
        """Convert to dict for API request."""
        return {
            "query_analysis_enabled": self.query_analysis_enabled,
            "enhanced_features_enabled": self.enhanced_features_enabled,
            "cot_enabled": self.cot_enabled,
            "reranker_enabled": self.reranker_enabled,
            "cove_mode": self.cove_mode,
        }


@dataclass
class ConfigResult:
    """Results for a single pipeline configuration."""

    config: PipelineConfig
    evaluation: BatchEvaluationResult
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

        # T-test on paired samples
        t_stat, p_value = stats.ttest_rel(paired_scores_a, paired_scores_b)

        # Effect size (Cohen's d for paired samples)
        # For paired t-test, use standard deviation of differences
        mean_diff = statistics.mean(paired_scores_a) - statistics.mean(paired_scores_b)
        differences = [a - b for a, b in zip(paired_scores_a, paired_scores_b)]
        std_diff = statistics.stdev(differences) if len(differences) > 1 else 0.0
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        return {
            "config_a": config_a,
            "config_b": config_b,
            "metric": metric,
            "mean_a": getattr(result_a.evaluation, metric),
            "mean_b": getattr(result_b.evaluation, metric),
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


class AblationStudy:
    """
    Run ablation study comparing different pipeline configurations.

    Uses RAGAS evaluator to score each configuration.
    """

    # Predefined configurations - 12 configs covering important toggle combinations

    # === Baseline (all off) ===
    BASELINE = PipelineConfig(
        name="baseline",
        description="No enhancements (vector search only)",
        query_analysis_enabled=False,
        enhanced_features_enabled=False,
        cot_enabled=False,
        reranker_enabled=False,
        cove_mode="off",
    )

    # === Single toggle configs ===
    QUERY_ONLY = PipelineConfig(
        name="query_only",
        description="Query analysis only (multihop detection)",
        query_analysis_enabled=True,
        enhanced_features_enabled=False,
        cot_enabled=False,
        reranker_enabled=False,
        cove_mode="off",
    )

    ENHANCED_ONLY = PipelineConfig(
        name="enhanced_only",
        description="Enhanced features only (metadata, quality checks)",
        query_analysis_enabled=False,
        enhanced_features_enabled=True,
        cot_enabled=False,
        reranker_enabled=False,
        cove_mode="off",
    )

    COT_ONLY = PipelineConfig(
        name="cot_only",
        description="Chain of Thought only",
        query_analysis_enabled=False,
        enhanced_features_enabled=False,
        cot_enabled=True,
        reranker_enabled=False,
        cove_mode="off",
    )

    RERANKER_ONLY = PipelineConfig(
        name="reranker_only",
        description="Reranker only",
        query_analysis_enabled=False,
        enhanced_features_enabled=False,
        cot_enabled=False,
        reranker_enabled=True,
        cove_mode="off",
    )

    COVE_AUTO_ONLY = PipelineConfig(
        name="cove_auto_only",
        description="CoVe auto-correction only",
        query_analysis_enabled=False,
        enhanced_features_enabled=False,
        cot_enabled=False,
        reranker_enabled=False,
        cove_mode="auto",
    )

    # === Important combinations ===
    COT_ENHANCED = PipelineConfig(
        name="cot_enhanced",
        description="CoT + Enhanced Features",
        query_analysis_enabled=False,
        enhanced_features_enabled=True,
        cot_enabled=True,
        reranker_enabled=False,
        cove_mode="off",
    )

    QUERY_RERANK = PipelineConfig(
        name="query_rerank",
        description="Query Analysis + Reranking",
        query_analysis_enabled=True,
        enhanced_features_enabled=False,
        cot_enabled=False,
        reranker_enabled=True,
        cove_mode="off",
    )

    # === CoVe mode variations ===
    FULL_COVE_AUTO = PipelineConfig(
        name="full_cove_auto",
        description="Full pipeline with CoVe auto-correction",
        query_analysis_enabled=True,
        enhanced_features_enabled=True,
        cot_enabled=True,
        reranker_enabled=True,
        cove_mode="auto",
    )

    FULL_COVE_METADATA = PipelineConfig(
        name="full_cove_metadata",
        description="Full pipeline with CoVe metadata-only",
        query_analysis_enabled=True,
        enhanced_features_enabled=True,
        cot_enabled=True,
        reranker_enabled=True,
        cove_mode="metadata",
    )

    FULL_COVE_SUGGEST = PipelineConfig(
        name="full_cove_suggest",
        description="Full pipeline with CoVe suggest mode",
        query_analysis_enabled=True,
        enhanced_features_enabled=True,
        cot_enabled=True,
        reranker_enabled=True,
        cove_mode="suggest",
    )

    # === Full (no CoVe) ===
    FULL_NO_COVE = PipelineConfig(
        name="full_no_cove",
        description="Full pipeline without CoVe",
        query_analysis_enabled=True,
        enhanced_features_enabled=True,
        cot_enabled=True,
        reranker_enabled=True,
        cove_mode="off",
    )

    def __init__(
        self,
        api_base_url: str,
        ragas_evaluator: Optional[RAGASEvaluator] = None,
    ):
        """
        Initialize ablation study.

        Args:
            api_base_url: Base URL for RAG API (e.g., http://localhost:8000)
            ragas_evaluator: RAGAS evaluator instance (creates default if None)
        """
        self.api_base_url = api_base_url.rstrip("/")
        self.ragas_evaluator = ragas_evaluator or RAGASEvaluator()

        # Configure retry strategy for network resilience
        retry_strategy = Retry(
            total=3,  # Max 3 retries
            backoff_factor=2,  # Wait 2s, 4s, 8s between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP codes
            allowed_methods=["POST"],  # Only retry POST requests
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info(f"Initialized ablation study with API: {self.api_base_url} (retry: 3x with backoff)")

    def __del__(self):
        """Clean up HTTP session on destruction."""
        if hasattr(self, 'session'):
            try:
                self.session.close()
                logger.debug("Closed HTTP session")
            except Exception as e:
                logger.warning(f"Error closing session: {e}")

    def run(
        self,
        questions_path: Path,
        configs: Optional[List[PipelineConfig]] = None,
        max_questions: Optional[int] = None,
    ) -> AblationStudyResult:
        """
        Run ablation study on given questions.

        Args:
            questions_path: Path to .jsonl file with test questions
            configs: List of configurations to test (uses default set if None)
            max_questions: Limit number of questions (for testing)

        Returns:
            AblationStudyResult with all metrics
        """
        # Load questions
        questions = self._load_questions(questions_path, max_questions)
        logger.info(f"Loaded {len(questions)} test questions")

        # Default configs if not provided (12 configs)
        if configs is None:
            configs = [
                self.BASELINE,
                self.QUERY_ONLY,
                self.ENHANCED_ONLY,
                self.COT_ONLY,
                self.RERANKER_ONLY,
                self.COVE_AUTO_ONLY,
                self.COT_ENHANCED,
                self.QUERY_RERANK,
                self.FULL_NO_COVE,
                self.FULL_COVE_AUTO,
                self.FULL_COVE_METADATA,
                self.FULL_COVE_SUGGEST,
            ]

        logger.info(f"Testing {len(configs)} configurations")

        # Run each configuration
        study_start = time.time()
        config_results = []

        for config in configs:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Running configuration: {config.name}")
            logger.info(f"Description: {config.description}")
            logger.info(f"{'=' * 80}\n")

            result = self._run_config(config, questions)
            config_results.append(result)

            logger.info(f"✓ {config.name} completed in {result.run_time_ms:.0f}ms")
            logger.info(f"  Faithfulness: {result.evaluation.mean_faithfulness:.3f}")
            logger.info(f"  Answer Relevancy: {result.evaluation.mean_answer_relevancy:.3f}")
            logger.info(f"  Latency: {result.evaluation.mean_latency_ms:.0f}ms")

        total_time_ms = (time.time() - study_start) * 1000

        return AblationStudyResult(
            config_results=config_results,
            questions=questions,
            num_questions=len(questions),
            total_time_ms=total_time_ms,
        )

    def _run_config(
        self,
        config: PipelineConfig,
        questions: List[Dict[str, Any]],
    ) -> ConfigResult:
        """Run single configuration on all questions."""
        start_time = time.time()

        # Collect RAG responses
        rag_questions = []
        rag_answers = []
        rag_contexts_list = []
        ground_truths = []
        metadata_list = []
        api_responses = []

        for i, q in enumerate(questions):
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{len(questions)}")

            try:
                # Call RAG API with config
                response = self._call_rag_api(
                    query=q["question"],
                    config=config,
                )

                # Validate response has required fields
                if not isinstance(response, dict):
                    raise ValueError(f"Invalid response type: {type(response)}")

                required_fields = ["answer", "contexts"]
                missing_fields = [f for f in required_fields if f not in response]
                if missing_fields:
                    raise ValueError(f"Missing required fields in API response: {missing_fields}")

                api_responses.append(response)

                # Extract data for RAGAS with safe defaults
                rag_questions.append(q["question"])
                rag_answers.append(response.get("answer", ""))
                rag_contexts_list.append(response.get("contexts", []))
                ground_truths.append(q["ground_truth"])

                # Metadata for custom metrics
                # Use response["sources"] which includes merged CoVe evidences
                sources_urls = [s.get("url") for s in response.get("sources", []) if s.get("url")]

                metadata = {
                    "latency_ms": response.get("metadata", {}).get("total_time_ms", 0.0),
                    "sources": sources_urls,  # Use merged sources (includes CoVe recovery)
                    "is_multihop": response.get("metadata", {}).get("is_multihop", False),
                    "sub_queries": response.get("sub_queries", []),
                    "query_type": response.get("metadata", {}).get("query_type"),
                }
                metadata_list.append(metadata)

            except Exception as e:
                logger.error(f"Failed to process question {i}: {e}")
                # Add placeholder to maintain alignment
                rag_questions.append(q["question"])
                rag_answers.append("")
                rag_contexts_list.append([])
                ground_truths.append(q["ground_truth"])
                metadata_list.append({})

        # Evaluate with RAGAS
        logger.info(f"Evaluating with RAGAS...")
        evaluation = self.ragas_evaluator.evaluate_batch(
            questions=rag_questions,
            answers=rag_answers,
            contexts_list=rag_contexts_list,
            ground_truths=ground_truths,
            metadata_list=metadata_list,
        )

        run_time_ms = (time.time() - start_time) * 1000

        return ConfigResult(
            config=config,
            evaluation=evaluation,
            run_time_ms=run_time_ms,
            api_responses=api_responses,
        )

    def _call_rag_api(
        self,
        query: str,
        config: PipelineConfig,
    ) -> Dict[str, Any]:
        """
        Call RAG API with specific configuration.

        Uses the /eval/ablation endpoint which supports toggling components.
        """
        url = f"{self.api_base_url}/eval/ablation"

        # Flatten config into request (no nested "config" object)
        payload = {
            "query": query,
            "query_analysis_enabled": config.query_analysis_enabled,
            "enhanced_features_enabled": config.enhanced_features_enabled,
            "cot_enabled": config.cot_enabled,
            "reranker_enabled": config.reranker_enabled,
            "cove_mode": config.cove_mode,
            "prompt_template": "auto",  # Auto-select based on query
            "top_k": 8,  # Standard top-k
        }

        response = self.session.post(url, json=payload, timeout=120)
        response.raise_for_status()

        return response.json()

    def _load_questions(
        self,
        path: Path,
        max_questions: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load test questions from .jsonl file."""
        questions = []

        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    question = json.loads(line)
                    questions.append(question)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue

                if max_questions and len(questions) >= max_questions:
                    break

        if not questions:
            raise ValueError(f"No valid questions loaded from {path}")

        return questions

    def save_results(
        self,
        result: AblationStudyResult,
        output_path: Path,
    ) -> None:
        """Save ablation study results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved results to {output_path}")
