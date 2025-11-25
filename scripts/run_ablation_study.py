from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

from src.ragx.evaluation.ablation_study import AblationStudy
from src.ragx.evaluation.ragas_evaluator import RAGASEvaluator
from src.ragx.evaluation.configs import (
    BASELINE,
    QUERY_ONLY,
    RERANKER_ONLY,
    COVE_AUTO_ONLY,
    FULL_NO_COVE,
    FULL_COVE_AUTO,
    FULL_COVE_METADATA,
    FULL_COVE_SUGGEST,
)

from colorlog import ColoredFormatter

formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s%(reset)s │ %(cyan)s%(name)s%(reset)s │ %(log_color)s%(levelname)-8s%(reset)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", reset=True,
    log_colors={'DEBUG': 'blue', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white', })

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)


def print_summary(result):
    """Print human-readable summary of results."""
    print(f"\n{'=' * 80}")
    print("ABLATION STUDY RESULTS")
    print(f"{'=' * 80}\n")

    print(f"Total Questions: {result.num_questions}")
    print(f"Total Time: {result.total_time_ms / 1000:.1f}s")
    print(f"Configurations Tested: {len(result.config_results)}\n")

    # Table header
    print(f"{'Configuration':<20} {'Faith':>7} {'Rel':>7} {'Prec':>7} {'Recall':>7} {'Latency':>9} {'Cands':>7} {'Srcs':>6} {'Cov':>6}")
    print(f"{'-' * 20} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 7} {'-' * 6} {'-' * 6}")

    # Results per config
    for cr in result.config_results:
        ev = cr.evaluation
        print(
            f"{cr.config.name:<20} "
            f"{ev.mean_faithfulness:>7.3f} "
            f"{ev.mean_answer_relevancy:>7.3f} "
            f"{ev.mean_context_precision:>7.3f} "
            f"{ev.mean_context_recall:>7.3f} "
            f"{ev.mean_latency_ms:>8.0f}ms "
            f"{ev.mean_num_candidates:>7.1f} "
            f"{ev.mean_sources_count:>6.1f} "
            f"{ev.mean_multihop_coverage:>6.3f}"
        )

    # Pipeline details
    print(f"\n{'=' * 80}")
    print("PIPELINE DETAILS")
    print(f"{'=' * 80}\n")

    for cr in result.config_results:
        ev = cr.evaluation
        cfg = cr.config

        print(f"Configuration: {cr.config.name}")
        print(f"  Description: {cr.config.description}")
        print(f"  Query Analysis: {'✓' if cfg.query_analysis_enabled else '✗'} | "
              f"Reranker: {'✓' if cfg.reranker_enabled else '✗'} | "
              f"CoT: {'✓' if cfg.cot_enabled else '✗'} | "
              f"CoVe: {cfg.cove_mode}")
        print(f"  Questions: {ev.num_questions} ({ev.num_multihop} multihop, {ev.num_simple} simple)")
        print(f"  Retrieval: {ev.mean_num_candidates:.1f} candidates → {ev.mean_sources_count:.1f} final sources")

        if ev.num_multihop > 0:
            print(f"  Multihop Coverage: {ev.mean_multihop_coverage:.1%} of sub-queries retrieved docs")

        # Show pipeline flow
        if cfg.reranker_enabled and ev.num_multihop > 0:
            print(f"  Pipeline: Retrieve ({ev.mean_num_candidates:.0f}) → Local Rerank → Fusion → Global Rerank → Top-K ({ev.mean_sources_count:.0f})")
        elif cfg.reranker_enabled:
            print(f"  Pipeline: Retrieve ({ev.mean_num_candidates:.0f}) → Rerank → Top-K ({ev.mean_sources_count:.0f})")
        else:
            print(f"  Pipeline: Retrieve ({ev.mean_num_candidates:.0f}) → Top-K ({ev.mean_sources_count:.0f})")

        print()

    # Best configs
    print(f"\n{'=' * 80}")
    print("BEST CONFIGURATIONS")
    print(f"{'=' * 80}\n")

    metrics = [
        ("Faithfulness", "mean_faithfulness"),
        ("Answer Relevancy", "mean_answer_relevancy"),
        ("Context Precision", "mean_context_precision"),
        ("Context Recall", "mean_context_recall"),
    ]

    for metric_name, metric_key in metrics:
        best = result.get_best_config(metric_key)
        score = getattr(best.evaluation, metric_key)
        print(f"{metric_name:<20}: {best.config.name:<15} ({score:.3f})")

    # Statistical comparisons
    print(f"\n{'=' * 80}")
    print("STATISTICAL COMPARISONS (t-tests)")
    print(f"{'=' * 80}\n")

    # Metrics to compare
    comparison_metrics = [
        ("mean_faithfulness", "Faithfulness"),
        ("mean_answer_relevancy", "Answer Relevancy"),
        ("mean_context_precision", "Context Precision"),
        ("mean_context_recall", "Context Recall"),
    ]

    # Compare full_cove_auto vs baseline
    if any(cr.config.name == "full_cove_auto" for cr in result.config_results) and \
            any(cr.config.name == "baseline" for cr in result.config_results):

        print("Full (CoVe Auto) vs Baseline:")
        print("-" * 80)
        for metric_key, metric_name in comparison_metrics:
            comparison = result.compare_configs("full_cove_auto", "baseline", metric_key)
            print(f"\n{metric_name}:")
            print(f"  Full (CoVe Auto): {comparison['mean_a']:.3f}")
            print(f"  Baseline:         {comparison['mean_b']:.3f}")
            print(f"  Diff:             {comparison['mean_diff']:+.3f}")
            print(f"  p-value:          {comparison['p_value']:.4f} {'✓ SIGNIFICANT' if comparison['significant'] else '✗ not significant'}")
            print(f"  Effect:           {comparison['effect_size']} (d={comparison['cohens_d']:.2f})")
        print()

    # Compare full_cove_auto vs full_no_cove
    if any(cr.config.name == "full_cove_auto" for cr in result.config_results) and \
            any(cr.config.name == "full_no_cove" for cr in result.config_results):

        print("Full (CoVe Auto) vs Full (No CoVe):")
        print("-" * 80)
        for metric_key, metric_name in comparison_metrics:
            comparison = result.compare_configs("full_cove_auto", "full_no_cove", metric_key)
            print(f"\n{metric_name}:")
            print(f"  With CoVe:    {comparison['mean_a']:.3f}")
            print(f"  Without CoVe: {comparison['mean_b']:.3f}")
            print(f"  Diff:         {comparison['mean_diff']:+.3f}")
            print(f"  p-value:      {comparison['p_value']:.4f} {'✓ SIGNIFICANT' if comparison['significant'] else '✗ not significant'}")
            print(f"  Effect:       {comparison['effect_size']} (d={comparison['cohens_d']:.2f})")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study on RAG pipeline configurations"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Path to .jsonl file with test questions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ablation_study.json"),
        help="Output path for results (default: results/ablation_study.json)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="RAG API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions (for testing)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=[
            "baseline",
            "query_only",
            "reranker_only",
            "cove_auto_only",
            "full_no_cove",
            "full_cove_auto",
            "full_cove_metadata",
            "full_cove_suggest",
        ],
        default=None,
        help="Specific configurations to test (default: all)",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="api",
        choices=["api", "ollama", "huggingface"],
        help="LLM provider for RAGAS evaluation (default: api)",
    )
    parser.add_argument(
        "--ragas-batch-size",
        type=int,
        default=2,
        help="Mini-batch size for RAGAS evaluation (default: 2 for API, use 10+ for local H100)",
    )
    parser.add_argument(
        "--ragas-delay",
        type=float,
        default=2.0,
        help="Delay in seconds between RAGAS mini-batches (default: 2.0 for API, use 0 for local)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/checkpoint/ablation_study.json"),
        help="Directory for checkpoint files (enables auto-save/resume)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Unique run ID for checkpointing (auto-generated if not provided)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available (requires --checkpoint-dir and --run-id)",
    )

    args = parser.parse_args()

    # Check questions file exists
    if not args.questions.exists():
        logger.error(f"Questions file not found: {args.questions}")
        sys.exit(1)

    # Initialize RAGAS evaluator
    logger.info(f"Initializing RAGAS evaluator with provider: {args.llm_provider}")
    ragas_evaluator = RAGASEvaluator(llm_provider=args.llm_provider)

    # Initialize ablation study
    ablation = AblationStudy(
        api_base_url=args.api_url,
        ragas_evaluator=ragas_evaluator,
        checkpoint_dir=args.checkpoint_dir,
        ragas_batch_size=args.ragas_batch_size,
        ragas_delay=args.ragas_delay,
    )

    # Select configs
    configs = None
    if args.configs:
        config_map = {
            "baseline": BASELINE,
            "query_only": QUERY_ONLY,
            "reranker_only": RERANKER_ONLY,
            "cove_auto_only": COVE_AUTO_ONLY,
            "full_no_cove": FULL_NO_COVE,
            "full_cove_auto": FULL_COVE_AUTO,
            "full_cove_metadata": FULL_COVE_METADATA,
            "full_cove_suggest": FULL_COVE_SUGGEST,
        }
        configs = [config_map[name] for name in args.configs]

    # Run study
    logger.info("Starting ablation study...")
    logger.info(f"Questions: {args.questions}")
    logger.info(f"Max questions: {args.max_questions or 'all'}")
    if args.checkpoint_dir:
        logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
        logger.info(f"Resume: {args.resume}")
        if args.run_id:
            logger.info(f"Run ID: {args.run_id}")

    result = ablation.run(
        questions_path=args.questions,
        configs=configs,
        max_questions=args.max_questions,
        run_id=args.run_id,
        resume=args.resume,
    )

    # Save results
    ablation.save_results(result, args.output)

    # Print summary
    print_summary(result)

    logger.info("\n✓ Ablation study complete!")
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()