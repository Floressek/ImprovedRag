from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import statistics

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy import stats

logger = logging.getLogger(__name__)
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available - progress bars disabled. Install with: pip install tqdm")

from src.ragx.evaluation.ragas_evaluator import RAGASEvaluator, BatchEvaluationResult


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
    prompt_template: str = "auto"  # "basic", "enhanced", "auto"
    # LLM provider
    provider: Optional[str] = None  # "api", "ollama", "huggingface", or None (use default)
    # Retrieval parameters
    top_k: int = 10

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
            "top_k": self.top_k,
        }
        if self.provider:
            request["provider"] = self.provider
        return request


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
        cot_enabled=False,
        reranker_enabled=False,
        cove_mode="off",
        prompt_template="basic",
    )

    # === Single toggle configs ===
    QUERY_ONLY = PipelineConfig(
        name="query_only",
        description="Query analysis only (multihop detection)",
        query_analysis_enabled=True,
        cot_enabled=False,
        reranker_enabled=False,
        cove_mode="off",
        prompt_template="basic",
    )

    ENHANCED_ONLY = PipelineConfig(
        name="enhanced_only",
        description="Enhanced features only (metadata, quality checks)",
        query_analysis_enabled=False,
        cot_enabled=False,
        reranker_enabled=False,
        cove_mode="off",
        prompt_template="enhanced",
    )

    COT_ONLY = PipelineConfig(
        name="cot_only",
        description="Chain of Thought only",
        query_analysis_enabled=False,
        cot_enabled=True,
        reranker_enabled=False,
        cove_mode="off",
        prompt_template="basic",
    )

    RERANKER_ONLY = PipelineConfig(
        name="reranker_only",
        description="Reranker only",
        query_analysis_enabled=False,
        cot_enabled=False,
        reranker_enabled=True,
        cove_mode="off",
        prompt_template="basic",
    )

    MULTIHOP_ONLY = PipelineConfig(
        name="multihop_only",
        description="Multihop detection only",
        query_analysis_enabled=True,
        cot_enabled=True,
        reranker_enabled=True,
        cove_mode="off",
        prompt_template="multihop",
    )

    COVE_AUTO_ONLY = PipelineConfig(
        name="cove_auto_only",
        description="CoVe auto-correction only",
        query_analysis_enabled=False,
        cot_enabled=False,
        reranker_enabled=False,
        cove_mode="auto",
        prompt_template="basic",
    )

    # === Important combinations ===
    COT_ENHANCED = PipelineConfig(
        name="cot_enhanced",
        description="CoT + Enhanced Features",
        query_analysis_enabled=False,
        cot_enabled=True,
        reranker_enabled=False,
        cove_mode="off",
        prompt_template="enhanced",
    )

    QUERY_RERANK = PipelineConfig(
        name="query_rerank",
        description="Query Analysis + Reranking",
        query_analysis_enabled=True,
        cot_enabled=False,
        reranker_enabled=True,
        cove_mode="off",
        prompt_template="basic",
    )

    # === CoVe mode variations ===
    FULL_COVE_AUTO = PipelineConfig(
        name="full_cove_auto",
        description="Full pipeline with CoVe auto-correction",
        query_analysis_enabled=True,
        cot_enabled=True,
        reranker_enabled=True,
        cove_mode="auto",
        prompt_template="multihop",
    )

    FULL_COVE_METADATA = PipelineConfig(
        name="full_cove_metadata",
        description="Full pipeline with CoVe metadata-only",
        query_analysis_enabled=True,
        cot_enabled=True,
        reranker_enabled=True,
        cove_mode="metadata",
        prompt_template="multihop",
    )

    FULL_COVE_SUGGEST = PipelineConfig(
        name="full_cove_suggest",
        description="Full pipeline with CoVe suggest mode",
        query_analysis_enabled=True,
        cot_enabled=True,
        reranker_enabled=True,
        cove_mode="suggest",
        prompt_template="multihop",
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
    )

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
        self.api_base_url = api_base_url.rstrip("/")
        self.ragas_evaluator = ragas_evaluator or RAGASEvaluator()
        self.api_timeout = api_timeout
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.save_every_n_questions = save_every_n_questions
        self.ragas_batch_size = ragas_batch_size
        self.ragas_delay = ragas_delay

        # Create checkpoint directory if specified
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpoint dir: {self.checkpoint_dir}")

        # Configure retry strategy for network resilience
        retry_strategy = Retry(
            total=retry_total,
            backoff_factor=retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP codes
            allowed_methods=["POST"],  # Only retry POST requests
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info(
            f"Initialized ablation study with API: {self.api_base_url} "
            f"(timeout: {api_timeout}s, retry: {retry_total}x with backoff {retry_backoff}s, "
            f"RAGAS: batch_size={ragas_batch_size}, delay={ragas_delay}s)"
        )

    def __del__(self):
        """Clean up HTTP session on destruction."""
        if hasattr(self, 'session'):
            try:
                self.session.close()
                logger.debug("Closed HTTP session")
            except Exception as e:
                logger.warning(f"Error closing session: {e}")

    def _get_checkpoint_path(self, run_id: str) -> Path:
        """Get checkpoint file path for given run ID."""
        if not self.checkpoint_dir:
            raise ValueError("Checkpoint directory not set")
        return self.checkpoint_dir / f"checkpoint_{run_id}.json"

    def _save_checkpoint(
            self,
            run_id: str,
            questions: List[Dict[str, Any]],
            configs: List[PipelineConfig],
            config_results: List[ConfigResult],
            current_config_idx: int,
    ) -> None:
        """Save checkpoint state to disk."""
        if not self.checkpoint_dir:
            return

        checkpoint = CheckpointState(
            timestamp=datetime.now().isoformat(),
            completed_configs=[cr.config.name for cr in config_results],
            config_results=[
                {
                    "name": cr.config.name,
                    "description": cr.config.description,
                    "config": cr.config.to_dict(),
                    "evaluation": cr.evaluation.to_dict(),
                    "run_time_ms": cr.run_time_ms,
                }
                for cr in config_results
            ],
            questions=questions,
            current_config_idx=current_config_idx,
            total_configs=len(configs),
        )

        checkpoint_path = self._get_checkpoint_path(run_id)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, run_id: str) -> Optional[CheckpointState]:
        """Load checkpoint state from disk."""
        if not self.checkpoint_dir:
            return None

        checkpoint_path = self._get_checkpoint_path(run_id)
        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        logger.info(f"📂 Checkpoint loaded: {checkpoint_path}")
        return CheckpointState.from_dict(data)

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
        if resume and self.checkpoint_dir:
            checkpoint = self._load_checkpoint(run_id)
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

        logger.info(f"Testing {len(configs)} configurations (run_id: {run_id})")

        # Restore config_results from checkpoint
        config_results = []
        start_config_idx = 0
        if checkpoint:
            # Reconstruct ConfigResult objects from checkpoint
            for cr_dict in checkpoint.config_results:
                # Find matching config
                matching_config = next((c for c in configs if c.name == cr_dict["name"]), None)
                if not matching_config:
                    logger.warning(f"Config {cr_dict['name']} not found in current configs, skipping")
                    continue

                # Reconstruct BatchEvaluationResult
                eval_dict = cr_dict["evaluation"]
                evaluation = BatchEvaluationResult(
                    mean_faithfulness=eval_dict["mean_faithfulness"],
                    mean_answer_relevancy=eval_dict["mean_answer_relevancy"],
                    mean_context_precision=eval_dict["mean_context_precision"],
                    mean_context_recall=eval_dict["mean_context_recall"],
                    mean_latency_ms=eval_dict["mean_latency_ms"],
                    mean_sources_count=eval_dict["mean_sources_count"],
                    mean_multihop_coverage=eval_dict["mean_multihop_coverage"],
                    num_questions=eval_dict["num_questions"],
                    num_multihop=eval_dict["num_multihop"],
                    num_simple=eval_dict["num_simple"],
                )

                config_result = ConfigResult(
                    config=matching_config,
                    evaluation=evaluation,
                    run_time_ms=cr_dict["run_time_ms"],
                )
                config_results.append(config_result)

            start_config_idx = checkpoint.current_config_idx
            logger.info(f"Restored {len(config_results)} completed configs")

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
            if self.checkpoint_dir:
                self._save_checkpoint(run_id, questions, configs, config_results, config_idx + 1)

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
                    response = self._call_rag_api(
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
            **config.to_dict(),  # Use config's to_dict() method
        }

        logger.debug(f"Sending request to {url}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        response = self.session.post(url, json=payload, timeout=self.api_timeout)

        # Log response details before raising
        if not response.ok:
            logger.error(f"API returned {response.status_code}: {response.text}")

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
