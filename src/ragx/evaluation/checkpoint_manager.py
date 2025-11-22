"""Checkpoint management for resumable ablation studies."""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.ragx.evaluation.models import CheckpointState, PipelineConfig, ConfigResult, BatchEvaluationResult

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint save/load for resumable ablation studies."""

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files (None = disabled)
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpoint dir: {self.checkpoint_dir}")

    def is_enabled(self) -> bool:
        """Check if checkpointing is enabled."""
        return self.checkpoint_dir is not None

    def get_checkpoint_path(self, run_id: str) -> Path:
        """
        Get checkpoint file path for given run ID.

        Args:
            run_id: Unique run identifier

        Returns:
            Path to checkpoint file

        Raises:
            ValueError: If checkpointing is not enabled
        """
        if not self.checkpoint_dir:
            raise ValueError("Checkpoint directory not set")
        return self.checkpoint_dir / f"checkpoint_{run_id}.json"

    def save(
            self,
            run_id: str,
            questions: List[Dict[str, Any]],
            configs: List[PipelineConfig],
            config_results: List[ConfigResult],
            current_config_idx: int,
    ) -> None:
        """
        Save checkpoint state to disk.

        Args:
            run_id: Unique run identifier
            questions: List of test questions
            configs: List of all pipeline configs
            config_results: List of completed config results
            current_config_idx: Index of next config to process
        """
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

        checkpoint_path = self.get_checkpoint_path(run_id)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

    def load(self, run_id: str) -> Optional[CheckpointState]:
        """
        Load checkpoint state from disk.

        Args:
            run_id: Unique run identifier

        Returns:
            CheckpointState if found, None otherwise
        """
        if not self.checkpoint_dir:
            return None

        checkpoint_path = self.get_checkpoint_path(run_id)
        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        logger.info(f"📂 Checkpoint loaded: {checkpoint_path}")
        return CheckpointState.from_dict(data)

    def restore_config_results(
            self,
            checkpoint: CheckpointState,
            configs: List[PipelineConfig],
    ) -> List[ConfigResult]:
        """
        Reconstruct ConfigResult objects from checkpoint data.

        Args:
            checkpoint: Loaded checkpoint state
            configs: List of current pipeline configs

        Returns:
            List of restored ConfigResult objects
        """
        config_results = []

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
                mean_num_candidates=eval_dict["mean_num_candidates"],
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

        logger.info(f"Restored {len(config_results)} completed configs")
        return config_results
