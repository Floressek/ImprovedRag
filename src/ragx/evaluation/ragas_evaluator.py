"""RAGAS-based evaluator for RAG pipelines."""
from __future__ import annotations

import logging
import time
from typing import List, Dict, Any, Optional

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
from src.ragx.evaluation.models import EvaluationResult, BatchEvaluationResult
from src.ragx.evaluation.metrics import (
    count_sources,
    calculate_multihop_coverage,
    safe_std,
    calculate_ci,
)

logger = logging.getLogger(__name__)


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

            except (APIError, APIConnectionError, TimeoutError) as e:
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

        # Use num_sources from API metadata (includes CoVe evidences)
        sources_count = metadata.get("num_sources", 0)
        # Fallback: count unique URLs if num_sources not provided
        if sources_count == 0:
            sources = metadata.get("sources") or []
            sources_count = count_sources(sources)

        # Retrieved candidates before reranking
        # For multihop: show average per sub-query instead of total sum
        sub_queries = metadata.get("sub_queries") or []
        results_by_subquery = metadata.get("results_by_subquery") or {}

        num_candidates = metadata.get("num_candidates", 0)
        if len(sub_queries) > 1 and results_by_subquery:
            # Multihop: calculate average candidates per sub-query
            total_candidates = sum(len(results) for results in results_by_subquery.values())
            num_candidates = total_candidates / len(sub_queries) if sub_queries else num_candidates

        multihop_coverage = calculate_multihop_coverage(
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
            num_candidates=num_candidates,
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
            mini_batch_size: int = 2,
            delay_between_batches: float = 2.0,
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple question-answer pairs.

        Args:
            questions: List of user questions
            answers: List of RAG system's answers
            contexts_list: List of retrieved contexts (per question)
            ground_truths: List of expected correct answers
            metadata_list: Optional metadata per question
            mini_batch_size: Number of questions per mini-batch (default: 2) to avoid rate limits
            delay_between_batches: Delay in seconds between mini-batches (default: 2.0)

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

        logger.info(f"Evaluating batch of {len(questions)} questions with mini-batch size {mini_batch_size}...")

        # Process in mini-batches to avoid rate limits
        all_ragas_results = []
        num_batches = (len(questions) + mini_batch_size - 1) // mini_batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * mini_batch_size
            end_idx = min(start_idx + mini_batch_size, len(questions))

            logger.info(f"Processing mini-batch {batch_idx + 1}/{num_batches} (questions {start_idx + 1}-{end_idx})...")

            # Prepare dataset for this mini-batch
            batch_dataset = Dataset.from_dict({
                "question": questions[start_idx:end_idx],
                "answer": answers[start_idx:end_idx],
                "contexts": contexts_list[start_idx:end_idx],
                "ground_truth": ground_truths[start_idx:end_idx],
            })

            # Run RAGAS evaluation on mini-batch
            start_time = time.time()
            batch_result = evaluate(
                batch_dataset,
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
            logger.info(f"Mini-batch {batch_idx + 1} completed in {eval_time:.0f}ms")

            all_ragas_results.append(batch_result.to_pandas())

            # Delay between batches (except after last batch)
            if batch_idx < num_batches - 1:
                logger.debug(f"Waiting {delay_between_batches}s before next mini-batch...")
                time.sleep(delay_between_batches)

        # Combine all mini-batch results
        import pandas as pd
        ragas_df = pd.concat(all_ragas_results, ignore_index=True)
        logger.info(f"All mini-batches completed. Total questions evaluated: {len(ragas_df)}")

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

            # Use num_sources from API metadata (includes CoVe evidences)
            sources_count = metadata_list[i].get("num_sources", 0)
            # Fallback: count unique URLs if num_sources not provided
            if sources_count == 0:
                sources = metadata_list[i].get("sources") or []
                sources_count = count_sources(sources)

            # Retrieved candidates before reranking
            # For multihop: show average per sub-query instead of total sum
            sub_queries = metadata_list[i].get("sub_queries") or []
            results_by_subquery = metadata_list[i].get("results_by_subquery") or {}

            num_candidates = metadata_list[i].get("num_candidates", 0)
            if len(sub_queries) > 1 and results_by_subquery:
                # Multihop: calculate average candidates per sub-query
                total_candidates = sum(
                    r if isinstance(r, int) else len(r)
                    for r in results_by_subquery.values()
                )
                num_candidates = total_candidates / len(sub_queries) if sub_queries else num_candidates

            multihop_coverage = calculate_multihop_coverage(
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
                num_candidates=num_candidates,
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

        # Defensive division (results is never empty due to validation above, but be safe)
        num_results = len(results) if results else 1

        return BatchEvaluationResult(
            mean_faithfulness=safe_float(ragas_df["faithfulness"].mean()),
            mean_answer_relevancy=safe_float(ragas_df["answer_relevancy"].mean()),
            mean_context_precision=safe_float(ragas_df["context_precision"].mean()),
            mean_context_recall=safe_float(ragas_df["context_recall"].mean()),
            mean_latency_ms=sum(r.latency_ms for r in results) / num_results if results else 0.0,
            mean_sources_count=sum(r.sources_count for r in results) / num_results if results else 0.0,
            mean_num_candidates=sum(r.num_candidates for r in results) / num_results if results else 0.0,
            mean_multihop_coverage=sum(r.multihop_coverage for r in results) / num_results if results else 0.0,
            ci_faithfulness=calculate_ci(faithfulness_vals),
            ci_answer_relevancy=calculate_ci(answer_rel_vals),
            ci_context_precision=calculate_ci(context_prec_vals),
            ci_context_recall=calculate_ci(context_rec_vals),
            std_faithfulness=safe_std(faithfulness_vals),
            std_answer_relevancy=safe_std(answer_rel_vals),
            std_context_precision=safe_std(context_prec_vals),
            std_context_recall=safe_std(context_rec_vals),
            results=results,
            num_questions=len(results),
            num_multihop=num_multihop,
            num_simple=num_simple,
        )
