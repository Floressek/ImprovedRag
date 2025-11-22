"""Ablation study runner for comparing pipeline configurations."""
from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available - progress bars disabled. Install with: pip install tqdm")

from src.ragx.evaluation.ragas_evaluator import RAGASEvaluator
from src.ragx.evaluation.models import (
    PipelineConfig,
    ConfigResult,
    AblationStudyResult,
)
from src.ragx.evaluation.configs import get_all_configs
from src.ragx.evaluation.checkpoint_manager import CheckpointManager
from src.ragx.evaluation.api_client import RAGAPIClient


class AblationStudy:
    """
    Run ablation study comparing different pipeline configurations.

    Uses RAGAS evaluator to score each configuration.
    """

    def __init__(
            self,
            api_base_url: str,
            ragas_evaluator: Optional[RAGASEvaluator] = None,
            api_timeout: int = 120,
            retry_total: int = 3,
            retry_backoff: int = 2,
            checkpoint_dir: Optional[Path] = None,
            save_every_n_questions: int = 10,
            ragas_batch_size: int = 2,
            ragas_delay: float = 2.0,
    ):
        """
        Initialize ablation study.

        Args:
            api_base_url: Base URL for RAG API (e.g., http://localhost:8000)
            ragas_evaluator: RAGAS evaluator instance (creates default if None)
            api_timeout: API request timeout in seconds (default: 120)
            retry_total: Max number of retries for failed requests (default: 3)
            retry_backoff: Backoff factor for retries in seconds (default: 2)
            checkpoint_dir: Directory for checkpoint files (default: None = disabled)
            save_every_n_questions: Save partial results every N questions (default: 10)
            ragas_batch_size: Mini-batch size for RAGAS evaluation (default: 2)
            ragas_delay: Delay in seconds between RAGAS mini-batches (default: 2.0)
        """
        self.ragas_evaluator = ragas_evaluator or RAGASEvaluator()
        self.ragas_batch_size = ragas_batch_size
        self.ragas_delay = ragas_delay
        self.save_every_n_questions = save_every_n_questions

        # Initialize API client
        self.api_client = RAGAPIClient(
            api_base_url=api_base_url,
            timeout=api_timeout,
            retry_total=retry_total,
            retry_backoff=retry_backoff,
        )

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        logger.info(
            f"Initialized ablation study with API: {api_base_url} "
            f"(RAGAS: batch_size={ragas_batch_size}, delay={ragas_delay}s)"
        )

    def run(
            self,
            questions_path: Path,
            configs: Optional[List[PipelineConfig]] = None,
            max_questions: Optional[int] = None,
            run_id: Optional[str] = None,
            resume: bool = False,
    ) -> AblationStudyResult:
        """
        Run ablation study on given questions with checkpoint support.

        Args:
            questions_path: Path to .jsonl file with test questions
            configs: List of configurations to test (uses default set if None)
            max_questions: Limit number of questions (for testing)
            run_id: Unique ID for this run (for checkpointing)
            resume: If True, resume from checkpoint if available

        Returns:
            AblationStudyResult with all metrics
        """
        # Generate run_id if not provided
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Try to load checkpoint if resume=True
        checkpoint = None
        if resume and self.checkpoint_manager.is_enabled():
            checkpoint = self.checkpoint_manager.load(run_id)
            if checkpoint:
                logger.info(f"🔄 Resuming from checkpoint ({len(checkpoint.completed_configs)}/{checkpoint.total_configs} configs completed)")
                questions = checkpoint.questions
            else:
                logger.info("No checkpoint found, starting fresh")

        # Load questions if not from checkpoint
        if checkpoint is None:
            questions = self._load_questions(questions_path, max_questions)
            logger.info(f"Loaded {len(questions)} test questions")

        # Default configs if not provided (12 configs)
        if configs is None:
            configs = get_all_configs()

        logger.info(f"Testing {len(configs)} configurations (run_id: {run_id})")

        # Restore config_results from checkpoint
        config_results = []
        start_config_idx = 0
        if checkpoint:
            config_results = self.checkpoint_manager.restore_config_results(checkpoint, configs)
            start_config_idx = checkpoint.current_config_idx

        # Run each configuration
        study_start = time.time()

        # Progress bar for configs
        configs_iterator = (
            tqdm(configs[start_config_idx:], desc="Configs", unit="config", initial=start_config_idx, total=len(configs))
            if TQDM_AVAILABLE
            else configs[start_config_idx:]
        )

        for config_idx, config in enumerate(configs_iterator if TQDM_AVAILABLE else configs[start_config_idx:], start=start_config_idx):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Running configuration [{config_idx + 1}/{len(configs)}]: {config.name}")
            logger.info(f"Description: {config.description}")
            logger.info(f"{'=' * 80}\n")

            result = self._run_config(config, questions, run_id=run_id)
            config_results.append(result)

            logger.info(f"✓ {config.name} completed in {result.run_time_ms:.0f}ms")
            logger.info(f"  Faithfulness: {result.evaluation.mean_faithfulness:.3f}")
            logger.info(f"  Answer Relevancy: {result.evaluation.mean_answer_relevancy:.3f}")
            logger.info(f"  Latency: {result.evaluation.mean_latency_ms:.0f}ms")

            # Save checkpoint after each config
            if self.checkpoint_manager.is_enabled():
                self.checkpoint_manager.save(run_id, questions, configs, config_results, config_idx + 1)

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
            run_id: Optional[str] = None,
    ) -> ConfigResult:
        """Run single configuration on all questions with progress tracking."""
        start_time = time.time()

        # Collect RAG responses
        rag_questions = []
        rag_answers = []
        rag_contexts_list = []
        ground_truths = []
        metadata_list = []
        api_responses = []

        # Error tracking
        failed_questions = []
        retry_count = 0

        # Progress bar for questions
        questions_iterator = (
            tqdm(questions, desc=f"  Questions ({config.name})", unit="q", leave=False)
            if TQDM_AVAILABLE
            else questions
        )

        # Debug: Check what we're iterating over
        logger.debug(f"Questions type: {type(questions)}, length: {len(questions)}")
        if questions:
            logger.debug(f"First question type: {type(questions[0])}, value: {questions[0]}")

        for i, q in enumerate(questions_iterator):
            max_retries = 3
            retry_delay = 2  # seconds

            for attempt in range(max_retries):
                try:
                    # Call RAG API with config
                    response = self.api_client.call_ablation_endpoint(
                        query=q["question"],
                        config=config,
                    )

                    # Validate response has required fields
                    if not isinstance(response, dict):
                        raise ValueError(f"Invalid response type: {type(response)}")

                    required_fields = ["answer", "sources"]
                    missing_fields = [f for f in required_fields if f not in response]
                    if missing_fields:
                        raise ValueError(f"Missing required fields in API response: {missing_fields}")

                    api_responses.append(response)

                    # Extract data for RAGAS with safe defaults
                    rag_questions.append(q["question"])
                    rag_answers.append(response.get("answer", ""))

                    # Extract text from sources for RAGAS (RAGAS needs List[str], not List[Dict])
                    sources = response.get("sources", [])
                    contexts = [s.get("text", "") for s in sources]
                    rag_contexts_list.append(contexts)
                    ground_truths.append(q["ground_truth"])

                    # Metadata for custom metrics
                    # Use response["sources"] which includes merged CoVe evidences
                    sources_urls = [s.get("url") for s in sources if s.get("url")]
                    response_metadata = response.get("metadata", {})

                    metadata = {
                        "latency_ms": response_metadata.get("total_time_ms", 0.0),
                        "sources": sources_urls,  # Use merged sources (includes CoVe recovery)
                        "num_candidates": response_metadata.get("num_candidates", 0),  # Retrieved before reranking
                        "num_sources": response_metadata.get("num_sources", len(sources)),  # Final sources count
                        "is_multihop": response_metadata.get("is_multihop", False),
                        "sub_queries": response_metadata.get("sub_queries", []),
                        "query_type": response_metadata.get("query_type"),
                        "results_by_subquery": response_metadata.get("results_by_subquery", {}),
                    }
                    metadata_list.append(metadata)

                    # Success - break retry loop
                    break

                except Exception as e:
                    retry_count += 1
                    if attempt < max_retries - 1:
                        logger.warning(f"Question {i+1} failed (attempt {attempt+1}/{max_retries}): {e}")
                        logger.warning(f"Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Question {i+1} failed after {max_retries} attempts: {e}")
                        failed_questions.append((i, q["question"], str(e)))
                        # Add placeholder to maintain alignment
                        rag_questions.append(q["question"])
                        rag_answers.append("")
                        rag_contexts_list.append([])
                        ground_truths.append(q["ground_truth"])
                        metadata_list.append({})

        # Error summary
        if failed_questions:
            logger.warning(f"\n{'=' * 80}")
            logger.warning(f"⚠️  {len(failed_questions)} questions failed after retries (total retries: {retry_count})")
            logger.warning(f"{'=' * 80}")
            for idx, question, error in failed_questions[:5]:  # Show first 5
                logger.warning(f"  [{idx+1}] {question[:60]}... → {error}")
            if len(failed_questions) > 5:
                logger.warning(f"  ... and {len(failed_questions) - 5} more")
            logger.warning(f"{'=' * 80}\n")

        # Evaluate with RAGAS
        logger.info(f"Evaluating with RAGAS...")
        evaluation = self.ragas_evaluator.evaluate_batch(
            questions=rag_questions,
            answers=rag_answers,
            contexts_list=rag_contexts_list,
            ground_truths=ground_truths,
            metadata_list=metadata_list,
            mini_batch_size=self.ragas_batch_size,
            delay_between_batches=self.ragas_delay,
        )

        run_time_ms = (time.time() - start_time) * 1000

        return ConfigResult(
            config=config,
            evaluation=evaluation,
            run_time_ms=run_time_ms,
            api_responses=api_responses,
        )

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
