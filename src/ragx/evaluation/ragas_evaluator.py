from __future__ import annotations

import logging
import math
import os
import statistics
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from openai import RateLimitError, APIError, APIConnectionError

from src.ragx.utils.settings import settings
from src.ragx.evaluation.langchain_adapters import LLMInferenceAdapter, EmbedderAdapter

logger = logging.getLogger(__name__)


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
    sources_count: int
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


class RAGASEvaluator:
    """
    Evaluate RAG pipeline using RAGAS framework + custom metrics.

    Official RAGAS Metrics:
    - faithfulness: Factual accuracy of answer vs contexts
    - answer_relevancy: How relevant answer is to question
    - context_precision: Precision of retrieved contexts
    - context_recall: Recall of retrieved contexts vs ground truth

    Custom Metrics:
    - latency_ms: Total pipeline latency
    - sources_count: Number of unique sources retrieved
    - multihop_coverage: For multihop queries, % of sub-queries with ≥1 doc
    """

    def __init__(
            self,
            llm_provider: str = "api",
            llm_temperature: float = 0.5,
            llm_max_tokens: int = 4092,
            embeddings_model: Optional[str] = None,
            embeddings_device: Optional[str] = None,
    ):
        """
        Initialize RAGAS evaluator with our LLM and embeddings infrastructure.

        Args:
            llm_provider: LLM provider ("api", "ollama", "huggingface")
            llm_temperature: Sampling temperature for LLM
            llm_max_tokens: Max tokens for LLM generation
            embeddings_model: SentenceTransformers model ID (uses settings if None)
            embeddings_device: Device for embeddings ("cpu", "cuda", "auto")
        """
        # Initialize our LangChain-compatible adapters
        self.llm = LLMInferenceAdapter(
            provider=llm_provider,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
        )
        self.embeddings = EmbedderAdapter(
            model_id=embeddings_model,
            device=embeddings_device,
        )

        # Configure RAGAS metrics with our LLM and embeddings
        # These are global singletons that need to be configured
        faithfulness.llm = self.llm
        answer_relevancy.llm = self.llm
        answer_relevancy.embeddings = self.embeddings
        context_precision.llm = self.llm
        context_recall.llm = self.llm

        logger.info(
            f"Initialized RAGAS evaluator with LLM provider: {llm_provider}, "
            f"Embeddings model: {self.embeddings.embedder.model_id}"
        )

    def evaluate_single(
            self,
            question: str,
            answer: str,
            contexts: List[str],
            ground_truth: str,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single question-answer pair.

        Args:
            question: User question
            answer: RAG system's answer
            contexts: Retrieved context chunks
            ground_truth: Expected correct answer
            metadata: Optional metadata (latency_ms, is_multihop, sub_queries, etc.)

        Returns:
            EvaluationResult with all metrics
        """
        metadata = metadata or {}

        # Prepare dataset for RAGAS
        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        })

        # Run RAGAS evaluation with retry on rate limits
        logger.debug(f"Evaluating question: {question[:50]}...")

        max_retries = 3
        retry_delay = 60  # seconds

        for attempt in range(max_retries):
            try:
                ragas_result = evaluate(
                    dataset,
                    metrics=[
                        faithfulness,
                        answer_relevancy,
                        context_precision,
                        context_recall,
                    ],
                    llm=self.llm,
                    embeddings=self.embeddings,
                )
                break  # Success, exit retry loop

            except RateLimitError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")
                    raise

            except (APIError, APIConnectionError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}, retrying in 30s...")
                    time.sleep(30)
                else:
                    logger.error(f"API error after {max_retries} attempts: {e}")
                    raise

        # Extract RAGAS scores
        ragas_scores = ragas_result.to_pandas().iloc[0]

        # Calculate custom metrics
        latency_ms = metadata.get("latency_ms", 0.0)
        sources = metadata.get("sources") or []
        sources_count = self._count_sources(sources)

        sub_queries = metadata.get("sub_queries") or []
        results_by_subquery = metadata.get("results_by_subquery") or {}
        multihop_coverage = self._calculate_multihop_coverage(
            sub_queries,
            results_by_subquery,
        )

        # Handle NaN values from RAGAS (replace with 0.0)
        def safe_float(value, default=0.0):
            try:
                f = float(value)
                return default if (f != f) else f  # NaN check
            except (ValueError, TypeError):
                return default

        return EvaluationResult(
            faithfulness=safe_float(ragas_scores["faithfulness"]),
            answer_relevancy=safe_float(ragas_scores["answer_relevancy"]),
            context_precision=safe_float(ragas_scores["context_precision"]),
            context_recall=safe_float(ragas_scores["context_recall"]),
            latency_ms=latency_ms,
            sources_count=sources_count,
            multihop_coverage=multihop_coverage,
            num_contexts=len(contexts),
            query_type=metadata.get("query_type"),
            is_multihop=metadata.get("is_multihop", False),
            num_sub_queries=len(sub_queries),
            details=metadata,
        )

    def evaluate_batch(
            self,
            questions: List[str],
            answers: List[str],
            contexts_list: List[List[str]],
            ground_truths: List[str],
            metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple question-answer pairs.

        Args:
            questions: List of user questions
            answers: List of RAG system's answers
            contexts_list: List of retrieved contexts (per question)
            ground_truths: List of expected correct answers
            metadata_list: Optional metadata per question

        Returns:
            BatchEvaluationResult with aggregated metrics
        """
        if metadata_list is None:
            metadata_list = [{}] * len(questions)

        # Validate input lists are not empty
        if not questions:
            raise ValueError("Cannot evaluate empty question list")

        if not (len(questions) == len(answers) == len(contexts_list) == len(ground_truths)):
            raise ValueError("All input lists must have the same length")

        if len(metadata_list) != len(questions):
            raise ValueError(
                f"metadata_list length ({len(metadata_list)}) must match "
                f"questions length ({len(questions)})"
            )

        logger.info(f"Evaluating batch of {len(questions)} questions...")

        # Prepare dataset for RAGAS
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        })

        # Run RAGAS evaluation (batch)
        # Limit parallel execution to avoid rate limits
        start_time = time.time()
        ragas_result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self.llm,
            embeddings=self.embeddings,
        )
        eval_time = (time.time() - start_time) * 1000
        logger.info(f"RAGAS evaluation completed in {eval_time:.0f}ms")

        # Convert to pandas for easier access
        ragas_df = ragas_result.to_pandas()

        # Helper to handle NaN
        def safe_float(value, default=0.0):
            try:
                f = float(value)
                return default if (f != f) else f  # NaN check
            except (ValueError, TypeError):
                return default

        # Calculate custom metrics per question
        results = []
        for i in range(len(questions)):
            latency_ms = metadata_list[i].get("latency_ms", 0.0)
            sources = metadata_list[i].get("sources") or []
            sources_count = self._count_sources(sources)

            sub_queries = metadata_list[i].get("sub_queries") or []
            results_by_subquery = metadata_list[i].get("results_by_subquery") or {}
            multihop_coverage = self._calculate_multihop_coverage(
                sub_queries,
                results_by_subquery,
            )

            result = EvaluationResult(
                faithfulness=safe_float(ragas_df.iloc[i]["faithfulness"]),
                answer_relevancy=safe_float(ragas_df.iloc[i]["answer_relevancy"]),
                context_precision=safe_float(ragas_df.iloc[i]["context_precision"]),
                context_recall=safe_float(ragas_df.iloc[i]["context_recall"]),
                latency_ms=latency_ms,
                sources_count=sources_count,
                multihop_coverage=multihop_coverage,
                num_contexts=len(contexts_list[i]),
                query_type=metadata_list[i].get("query_type"),
                is_multihop=metadata_list[i].get("is_multihop", False),
                num_sub_queries=len(sub_queries),
                details=metadata_list[i],
            )
            results.append(result)

        # Aggregate metrics
        num_multihop = sum(1 for r in results if r.is_multihop)
        num_simple = len(results) - num_multihop

        # Calculate confidence intervals and std devs
        faithfulness_vals = [r.faithfulness for r in results]
        answer_rel_vals = [r.answer_relevancy for r in results]
        context_prec_vals = [r.context_precision for r in results]
        context_rec_vals = [r.context_recall for r in results]

        # Defensive division (results is never empty due to line 299, but be safe)
        num_results = len(results) if results else 1

        return BatchEvaluationResult(
            mean_faithfulness=safe_float(ragas_df["faithfulness"].mean()),
            mean_answer_relevancy=safe_float(ragas_df["answer_relevancy"].mean()),
            mean_context_precision=safe_float(ragas_df["context_precision"].mean()),
            mean_context_recall=safe_float(ragas_df["context_recall"].mean()),
            mean_latency_ms=sum(r.latency_ms for r in results) / num_results if results else 0.0,
            mean_sources_count=sum(r.sources_count for r in results) / num_results if results else 0.0,
            mean_multihop_coverage=sum(r.multihop_coverage for r in results) / num_results if results else 0.0,
            ci_faithfulness=self._calculate_ci(faithfulness_vals),
            ci_answer_relevancy=self._calculate_ci(answer_rel_vals),
            ci_context_precision=self._calculate_ci(context_prec_vals),
            ci_context_recall=self._calculate_ci(context_rec_vals),
            std_faithfulness=self._safe_std(faithfulness_vals),
            std_answer_relevancy=self._safe_std(answer_rel_vals),
            std_context_precision=self._safe_std(context_prec_vals),
            std_context_recall=self._safe_std(context_rec_vals),
            results=results,
            num_questions=len(results),
            num_multihop=num_multihop,
            num_simple=num_simple,
        )

    def _count_sources(self, sources: List[str]) -> int:
        """Count unique sources (URLs, doc IDs, etc.)."""
        return len(set(sources)) if sources else 0

    def _calculate_multihop_coverage(
            self,
            sub_queries: List[str],
            results_by_subquery: Dict[str, List[Any]],
    ) -> float:
        """
        Calculate multihop coverage: ratio of sub-queries with ≥1 retrieved doc.

        For multihop queries, this shows how well the retrieval covers all sub-queries.
        A score of 1.0 means every sub-query got at least one document.
        A score of 0.5 means only half the sub-queries got documents.

        Args:
            sub_queries: List of sub-queries from decomposition
            results_by_subquery: Dict mapping sub-query → retrieved results

        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        if not sub_queries or len(sub_queries) <= 1:
            # Single query or no decomposition → N/A
            return 1.0

        covered = 0
        for sq in sub_queries:
            results = results_by_subquery.get(sq, [])
            if results and len(results) > 0:
                covered += 1

        return covered / len(sub_queries)

    def _safe_std(self, values: List[float]) -> float:
        """Calculate standard deviation, handling edge cases."""
        if len(values) < 2:
            return 0.0
        try:
            return statistics.stdev(values)
        except statistics.StatisticsError:
            return 0.0

    def _calculate_ci(
            self,
            values: List[float],
            confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval.

        Uses formula: CI = mean ± z * (std / sqrt(n))
        For 95% CI, z = 1.96

        Args:
            values: List of metric values
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(values) < 2:
            # Not enough data for CI - need at least 2 samples
            # Return (0, 0) to indicate invalid CI rather than misleading equal bounds
            logger.warning(f"Cannot calculate CI with n={len(values)} samples (need n>=2)")
            return (0.0, 0.0)

        try:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            n = len(values)

            # Z-score for 95% CI
            z_score = 1.96

            # Margin of error
            margin = z_score * (std_val / math.sqrt(n))

            return (mean_val - margin, mean_val + margin)

        except (statistics.StatisticsError, ZeroDivisionError):
            mean_val = statistics.mean(values) if values else 0.0
            return (mean_val, mean_val)