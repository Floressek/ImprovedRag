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
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import statistics

import requests
from scipy import stats

from src.ragx.evaluation.ragas_evaluator import RAGASEvaluator, BatchEvaluationResult
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a pipeline variant."""

    name: str
    description: str
    query_analysis_enabled: bool = True
    reranker_enabled: bool = True
    cove_enabled: bool = True
    multihop_enabled: bool = True

    def to_dict(self) -> Dict[str, bool]:
        """Convert to dict for API request."""
        return {
            "query_analysis_enabled": self.query_analysis_enabled,
            "reranker_enabled": self.reranker_enabled,
            "cove_enabled": self.cove_enabled,
            "multihop_enabled": self.multihop_enabled,
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
        """
        # Find configs
        result_a = next((cr for cr in self.config_results if cr.config.name == config_a), None)
        result_b = next((cr for cr in self.config_results if cr.config.name == config_b), None)

        if not result_a or not result_b:
            raise ValueError(f"Configuration not found: {config_a} or {config_b}")

        # Extract per-question scores
        scores_a = [getattr(r, metric.replace("mean_", "")) for r in result_a.evaluation.results]
        scores_b = [getattr(r, metric.replace("mean_", "")) for r in result_b.evaluation.results]

        # T-test
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        # Effect size (Cohen's d)
        mean_diff = statistics.mean(scores_a) - statistics.mean(scores_b)
        pooled_std = statistics.stdev(scores_a + scores_b)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

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

    # Predefined configurations
    BASELINE = PipelineConfig(
        name="baseline",
        description="No enhancements (vector search only)",
        query_analysis_enabled=False,
        reranker_enabled=False,
        cove_enabled=False,
        multihop_enabled=False,
    )

    QUERY_ONLY = PipelineConfig(
        name="query_only",
        description="Query analysis/rewriting only",
        query_analysis_enabled=True,
        reranker_enabled=False,
        cove_enabled=False,
        multihop_enabled=True,  # Multihop is part of query analysis
    )

    RERANKER_ONLY = PipelineConfig(
        name="reranker_only",
        description="Reranker only (no query analysis)",
        query_analysis_enabled=False,
        reranker_enabled=True,
        cove_enabled=False,
        multihop_enabled=False,
    )

    COVE_ONLY = PipelineConfig(
        name="cove_only",
        description="CoVe verification only",
        query_analysis_enabled=False,
        reranker_enabled=False,
        cove_enabled=True,
        multihop_enabled=False,
    )

    FULL = PipelineConfig(
        name="full",
        description="All enhancements enabled",
        query_analysis_enabled=True,
        reranker_enabled=True,
        cove_enabled=True,
        multihop_enabled=True,
    )

    NO_COVE = PipelineConfig(
        name="no_cove",
        description="Full pipeline without CoVe",
        query_analysis_enabled=True,
        reranker_enabled=True,
        cove_enabled=False,
        multihop_enabled=True,
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
        logger.info(f"Initialized ablation study with API: {self.api_base_url}")

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

        # Default configs if not provided
        if configs is None:
            configs = [
                self.BASELINE,
                self.QUERY_ONLY,
                self.RERANKER_ONLY,
                self.COVE_ONLY,
                self.NO_COVE,
                self.FULL,
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

                api_responses.append(response)

                # Extract data for RAGAS
                rag_questions.append(q["question"])
                rag_answers.append(response["answer"])
                rag_contexts_list.append(response["contexts"])
                ground_truths.append(q["ground_truth"])

                # Metadata for custom metrics
                metadata = {
                    "latency_ms": response.get("metadata", {}).get("total_time_ms", 0.0),
                    "sources": [ctx.get("url") for ctx in response.get("context_details", [])],
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

        payload = {
            "query": query,
            "config": config.to_dict(),
            "top_k": 5,  # Standard top-k
        }

        response = requests.post(url, json=payload, timeout=120)
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
            for line in f:
                questions.append(json.loads(line.strip()))

                if max_questions and len(questions) >= max_questions:
                    break

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
