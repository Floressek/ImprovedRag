from __future__ import annotations

import logging
import json
import random
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from tqdm import tqdm

from src.ragx.generation.inference import LLMInference
from src.ragx.retrieval.rerankers.reranker import Reranker

logger = logging.getLogger(__name__)


class WikipediaQuestionGenerator:
    """Generate test questions from Wikipedia .jsonl files."""

    QUESTION_TYPES = ["simple", "comparison", "multihop", "temporal"]

    def __init__(
            self,
            data_dir: Path,
            llm: Optional[LLMInference] = None,
            validate_grounding: bool = True,
            similarity_threshold: float = 0.5,
            prompts_path: Optional[Path] = None,
    ):
        """
        Initialize generator.

        Args:
            data_dir: Path to processed/wiki_extracted/
            llm: LLM for question generation
            validate_grounding: Whether to validate ground truth is in contexts using cross-encoder
            similarity_threshold: Min cross-encoder similarity score (0-1) for validation
            prompts_path: Path to YAML prompts file (default: generator/prompts/generation_prompt.yaml)
        """
        self.data_dir = Path(data_dir)
        self.llm = llm or LLMInference(provider="api")  # Use API provider for generation
        self.validate_grounding = validate_grounding
        self.similarity_threshold = similarity_threshold

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Initialize cross-encoder for semantic validation
        if self.validate_grounding:
            logger.info("Initializing cross-encoder for ground truth validation...")
            self.reranker = Reranker(show_progress=False)
        else:
            self.reranker = None

        # Load prompts from YAML
        if prompts_path is None:
            prompts_path = Path(__file__).parent / "prompts" / "generation_prompt.yaml"

        self.prompts_path = Path(prompts_path)
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_path}")

        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
        logger.info(f"Loaded prompts from {self.prompts_path}")

    def load_articles_sample(
            self,
            folders: List[str],
            sample_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Load random sample of articles using reservoir sampling (memory-efficient).

        Args:
            folders: List of folder names (e.g. ["AA", "AB", ...])
            sample_size: Number of articles to sample

        Returns:
            List of sampled article dicts
        """
        # Reservoir sampling: memory-efficient random sampling from stream
        reservoir = []
        total_seen = 0

        for folder in folders:
            folder_path = self.data_dir / folder

            if not folder_path.exists():
                logger.warning(f"Folder not found: {folder_path}")
                continue

            # Find all wiki_* files (WikiExtractor output format)
            wiki_files = list(folder_path.glob("wiki_*"))

            if not wiki_files:
                logger.warning(f"No wiki_* files in {folder_path}")
                continue

            for wiki_file in wiki_files:
                try:
                    with open(wiki_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            article = json.loads(line.strip())

                            # Basic validation
                            if all(k in article for k in ["id", "title", "text", "url"]):
                                # Skip very short articles
                                if len(article["text"]) > 200:
                                    total_seen += 1

                                    # Reservoir sampling algorithm
                                    if len(reservoir) < sample_size:
                                        reservoir.append(article)
                                    else:
                                        # Replace with decreasing probability
                                        rand_idx = random.randint(0, total_seen - 1)
                                        if rand_idx < sample_size:
                                            reservoir[rand_idx] = article
                except Exception as e:
                    logger.error(f"Failed to load {wiki_file}: {e}")

        logger.info(f"Sampled {len(reservoir)} articles from {total_seen} total (memory-efficient)")
        return reservoir

    def generate_questions(
            self,
            num_questions: int = 1000,
            folders: Optional[List[str]] = None,
            distribution: Optional[Dict[str, float]] = None,
            max_attempts_per_question: int = 10,
            sample_size_per_folder: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate EXACT number of validated questions from Wikipedia articles.

        Will keep retrying and loading more articles until target count is reached.

        Args:
            num_questions: EXACT number of questions to generate
            folders: Initial folder names (default: AA-AJ). Will load more if needed.
            distribution: Question type distribution (default: 40/25/25/10)
            max_attempts_per_question: Max consecutive failures before giving up
            sample_size_per_folder: Articles to load per folder

        Returns:
            List of exactly num_questions validated question dicts
        """
        if folders is None:
            # First 10 folders, change the magic number!
            folders = [f"{chr(65)}{chr(65 + i)}" for i in range(10)]  # AA, AB, ..., AJ

        if distribution is None:
            distribution = {
                "simple": 0.40,
                "comparison": 0.25,
                "multihop": 0.25,
                "temporal": 0.10,
            }

        # Calculate target count per type
        targets = {}
        for qtype, ratio in distribution.items():
            targets[qtype] = int(num_questions * ratio)

        total_assigned = sum(targets.values())
        if total_assigned < num_questions:
            first_type = list(targets.keys())[0]
            targets[first_type] += (num_questions - total_assigned)

        logger.info(f"Target distribution: {targets}")

        # Load initial article sample
        initial_sample_size = sample_size_per_folder * len(folders)
        logger.info(f"Loading initial {initial_sample_size} articles from {len(folders)} folders...")
        articles = self.load_articles_sample(folders, sample_size=initial_sample_size)
        logger.info(f"Loaded {len(articles)} articles")

        # Prepare list of all available folders for dynamic loading
        all_folders = []
        for i in range(26):  # A-Z
            for j in range(26):  # A-Z
                all_folders.append(f"{chr(65 + i)}{chr(65 + j)}")

        # Track which folders we've already loaded
        loaded_folders = set(folders)
        folder_idx = 0

        questions = []
        type_breakdown = {qtype: 0 for qtype in self.QUESTION_TYPES}

        for qtype, target_count in targets.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Generating {target_count} {qtype} questions...")
            logger.info(f"{'=' * 60}")

            generated_count = 0
            total_attempts = 0
            consecutive_failures = 0

            pbar = tqdm(total=target_count, desc=f"{qtype.capitalize()}", unit="q")

            while generated_count < target_count:
                # Check if we need more articles
                if total_attempts > 0 and total_attempts % 20 == 0:
                    if len(articles) < 50:  # Running low on articles
                        logger.info(f"\n Running low on articles ({len(articles)}), loading more...")

                        # Find next unloaded folder
                        loaded_new = False
                        while folder_idx < len(all_folders):
                            next_folder = all_folders[folder_idx]
                            folder_idx += 1

                            if next_folder not in loaded_folders:
                                folder_path = self.data_dir / next_folder
                                if folder_path.exists():
                                    logger.info(f"Loading articles from {next_folder}...")
                                    new_articles = self.load_articles_sample(
                                        [next_folder],
                                        sample_size=sample_size_per_folder
                                    )
                                    if new_articles:
                                        articles.extend(new_articles)
                                        loaded_folders.add(next_folder)
                                        logger.info(f"✓ Loaded {len(new_articles)} articles (total: {len(articles)})")
                                        loaded_new = True
                                        break

                        if not loaded_new:
                            logger.warning("No more folders available to load!")
                            if len(articles) == 0:
                                logger.error(
                                    f"Out of articles! Generated {generated_count}/{target_count} {qtype} questions")
                                break

                # Generate question
                try:
                    question = self._generate_question(qtype, articles)
                    total_attempts += 1

                    if question is None:
                        consecutive_failures += 1
                        if consecutive_failures >= max_attempts_per_question:
                            logger.warning(
                                f"Too many consecutive failures ({consecutive_failures}), stopping {qtype} generation. "
                                f"Generated {generated_count}/{target_count}"
                            )
                            break
                        continue

                    # Validate grounding
                    if self.validate_grounding:
                        if not self._validate_grounding(question["ground_truth"], question["contexts"]):
                            consecutive_failures += 1
                            logger.debug(f"Question failed validation (consecutive: {consecutive_failures})")
                            if consecutive_failures >= max_attempts_per_question:
                                logger.warning(
                                    f"Too many consecutive validation failures ({consecutive_failures}), stopping {qtype}. "
                                    f"Generated {generated_count}/{target_count}"
                                )
                                break
                            continue

                    # Success!
                    questions.append(question)
                    type_breakdown[qtype] += 1
                    generated_count += 1
                    consecutive_failures = 0  # Reset failure counter on success
                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Error generating {qtype} question: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_attempts_per_question:
                        logger.warning(
                            f"Too many consecutive errors ({consecutive_failures}), stopping {qtype}. "
                            f"Generated {generated_count}/{target_count}"
                        )
                        break
                    continue

            pbar.close()

            logger.info(
                f"Generated {generated_count}/{target_count} {qtype} questions "
                f"(total attempts: {total_attempts})"
            )

        random.shuffle(questions)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"GENERATION COMPLETE")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total questions: {len(questions)}/{num_questions}")
        logger.info(f"Question type breakdown:")
        for qtype in self.QUESTION_TYPES:
            count = type_breakdown[qtype]
            percentage = (count / len(questions) * 100) if questions else 0
            logger.info(f"  {qtype:12s}: {count:4d} ({percentage:5.1f}%)")
        logger.info(f"{'=' * 60}")

        return questions

    def _generate_question(
            self,
            qtype: str,
            articles: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Generate single question of given type."""

        if qtype == "simple":
            return self._generate_simple(articles)
        elif qtype == "comparison":
            return self._generate_comparison(articles)
        elif qtype == "multihop":
            return self._generate_multihop(articles)
        elif qtype == "temporal":
            return self._generate_temporal(articles)
        else:
            logger.warning(f"Unknown question type: {qtype}")
            return None

    def _generate_simple(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate simple factual question from 1 article."""

        article = random.choice(articles)
        text = article["text"][:1000]

        prompt_template = self.prompts["generate_simple"]["template"]
        prompt = prompt_template.format(
            title=article['title'],
            text=text
        )

        response = self.llm.generate(
            prompt=prompt,
            temperature=0.3,
            max_new_tokens=4092,
            chain_of_thought_enabled=False,
        )

        try:
            data = json.loads(response.strip())

            return {
                "question": data["question"],
                "ground_truth": data["ground_truth"],
                "type": "simple",
                "source_urls": [article["url"]],
                "contexts": [text],
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {response}")
            return None

    def _generate_comparison(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparison question from 2 articles."""

        article1, article2 = random.sample(articles, 2)
        text1 = article1["text"][:1500]
        text2 = article2["text"][:1500]

        # Load prompt from YAML
        prompt_template = self.prompts["generate_comparison"]["template"]
        prompt = prompt_template.format(
            title1=article1['title'],
            text1=text1,
            title2=article2['title'],
            text2=text2
        )

        response = self.llm.generate(
            prompt=prompt,
            temperature=0.3,
            max_new_tokens=4092,
            chain_of_thought_enabled=False,
        )

        try:
            data = json.loads(response.strip())
            return {
                "question": data["question"],
                "ground_truth": data["ground_truth"],
                "type": "comparison",
                "source_urls": [article1["url"], article2["url"]],
                "contexts": [text1, text2],
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {response}")
            return None

    def _generate_multihop(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate multihop reasoning question from 2-3 articles."""

        num_articles = random.choice([2, 3])
        selected = random.sample(articles, num_articles)

        texts = [a["text"][:1200] for a in selected]
        titles = [a["title"] for a in selected]

        articles_text = "\n\n".join([
            f"ARTICLE {i + 1} TITLE: {titles[i]}\nTEXT:\n{texts[i]}"
            for i in range(num_articles)
        ])

        prompt_template = self.prompts["generate_multihop"]["template"]
        prompt = prompt_template.format(articles_text=articles_text)

        response = self.llm.generate(
            prompt=prompt,
            temperature=0.3,
            max_new_tokens=4092,
            chain_of_thought_enabled=False,
        )

        try:
            data = json.loads(response.strip())
            return {
                "question": data["question"],
                "ground_truth": data["ground_truth"],
                "type": "multihop",
                "source_urls": [a["url"] for a in selected],
                "contexts": texts,
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {response}")
            return None

    def _generate_temporal(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate temporal/chronological question."""

        article = random.choice(articles)
        text = article["text"][:1500]

        # Load prompt from YAML
        prompt_template = self.prompts["generate_temporal"]["template"]
        prompt = prompt_template.format(
            title=article['title'],
            text=text
        )

        response = self.llm.generate(
            prompt=prompt,
            temperature=0.3,
            max_new_tokens=4092,
            chain_of_thought_enabled=False,
        )

        try:
            data = json.loads(response.strip())
            return {
                "question": data["question"],
                "ground_truth": data["ground_truth"],
                "type": "temporal",
                "source_urls": [article["url"]],
                "contexts": [text],
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {response}")
            return None

    def _validate_grounding(
            self,
            ground_truth: str,
            contexts: List[str],
    ) -> bool:
        """
        Validate that ground truth is grounded in contexts using cross-encoder.

        Uses semantic similarity: scores ground_truth against concatenated contexts.
        For multi-context questions (comparison, multihop), this ensures ground truth
        can reference information from multiple sources.

        Args:
            ground_truth: Expected answer
            contexts: Context chunks (may be from multiple articles)

        Returns:
            True if ground truth is sufficiently grounded (semantically similar to contexts)
        """
        if not contexts:
            logger.warning("No contexts provided for validation")
            return False

        if not ground_truth.strip():
            logger.warning("Empty ground truth provided")
            return False

        # Filter empty contexts
        valid_contexts = [ctx for ctx in contexts if ctx.strip()]
        if not valid_contexts:
            logger.warning("No valid contexts after filtering")
            return False

        # Concatenate all contexts with separator
        # This handles multihop/comparison where ground_truth references multiple articles
        combined_context = "\n\n".join(valid_contexts)

        # Score using cross-encoder (higher score = more similar/relevant)
        try:
            score = self.reranker.model.predict(
                [[ground_truth, combined_context]],
                batch_size=1,
                show_progress_bar=False,
            )[0]

            score = float(score)

            if score < self.similarity_threshold:
                logger.info(
                    f"Ground truth not sufficiently grounded: "
                    f"similarity {score:.3f} < threshold {self.similarity_threshold:.3f} "
                    f"({len(valid_contexts)} contexts)"
                )
                return False

            logger.info(
                f"Ground truth validated: similarity {score:.3f} >= threshold {self.similarity_threshold:.3f} "
                f"({len(valid_contexts)} contexts)"
            )
            return True

        except Exception as e:
            logger.error(f"Error during cross-encoder validation: {e}")
            return False

    def save_questions(
            self,
            questions: List[Dict[str, Any]],
            output_path: Path,
    ) -> None:
        """Save questions to .jsonl file."""

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(questions)} questions to {output_path}")
