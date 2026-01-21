from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ragx.evaluation.generator.wikipedia_generator import WikipediaQuestionGenerator
from src.ragx.generation.inference import LLMInference

from colorlog import ColoredFormatter

formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s%(reset)s │ %(cyan)s%(name)s%(reset)s │ %(log_color)s%(levelname)-8s%(reset)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", reset=True,
    log_colors={'DEBUG': 'blue', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white', })

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate test questions from Wikipedia articles"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=1000,
        help="Number of questions to generate (default: 1000)",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=None,
        help="Folder names to sample from (default: AA-AJ)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/wiki_extracted"),
        help="Path to wiki_extracted directory (default: data/processed/wiki_extracted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/generated_questions.jsonl"),
        help="Output path for questions (default: data/eval/generated_questions.jsonl)",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default=None,
        help='Question type distribution as JSON (e.g., \'{"simple": 0.4, "comparison": 0.25, "multihop": 0.25, "temporal": 0.1}\')',
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="api",
        help="LLM provider (default: api)",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=3,
        help="Number of sample questions to display (default: 3)",
    )

    args = parser.parse_args()

    # Parse distribution if provided
    distribution = None
    if args.distribution:
        try:
            distribution = json.loads(args.distribution)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid distribution JSON: {e}")
            sys.exit(1)

    # Initialize LLM
    logger.info(f"Initializing LLM with provider: {args.provider}")
    llm = LLMInference(provider=args.provider)
    # llm = LLMInference(provider="ollama")

    # Initialize generator
    logger.info(f"Initializing question generator with data_dir: {args.data_dir}")
    try:
        generator = WikipediaQuestionGenerator(
            data_dir=args.data_dir,
            llm=llm,
        )
    except FileNotFoundError as e:
        logger.error(f"Data directory not found: {e}")
        sys.exit(1)

    # Generate questions
    logger.info(f"Generating {args.num_questions} questions...")
    logger.info(f"Folders: {args.folders or 'AA-AJ (default)'}")

    try:
        questions = generator.generate_questions(
            num_questions=args.num_questions,
            folders=args.folders,
            distribution=distribution,
        )
    except Exception as e:
        logger.error(f"Failed to generate questions: {e}")
        raise

    # Save questions
    logger.info(f"Saving questions to {args.output}")
    generator.save_questions(questions, args.output)

    # Display samples
    if args.show_samples > 0 and questions:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"SAMPLE QUESTIONS (showing {min(args.show_samples, len(questions))}):")
        logger.info(f"{'=' * 80}\n")

        for i, q in enumerate(questions[:args.show_samples], 1):
            logger.info(f"--- Sample {i} ({q['type']}) ---")
            logger.info(f"Question: {q['question']}")
            logger.info(f"Ground Truth: {q['ground_truth']}")
            logger.info(f"Sources: {len(q['source_urls'])} URLs")
            logger.info(f"Contexts: {len(q['contexts'])} snippets")
            logger.info("")

    # Statistics
    type_counts = {}
    for q in questions:
        qtype = q.get("type", "unknown")
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    logger.info(f"\n{'=' * 80}")
    logger.info("GENERATION STATISTICS:")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total questions generated: {len(questions)}")
    logger.info("Question type breakdown:")
    for qtype, count in sorted(type_counts.items()):
        percentage = (count / len(questions)) * 100
        logger.info(f"  {qtype:12s}: {count:4d} ({percentage:5.1f}%)")
    logger.info(f"\nOutput saved to: {args.output}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
