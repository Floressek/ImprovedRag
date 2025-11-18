from __future__ import annotations

import logging
import json
import random
import re
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from src.ragx.generation.inference import LLMInference
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


class WikipediaQuestionGenerator:
    """Generate test questions from Wikipedia .jsonl files."""

    QUESTION_TYPES = ["simple", "comparison", "multihop", "temporal"]

    def __init__(
        self,
        data_dir: Path,
        llm: Optional[LLMInference] = None,
        validate_grounding: bool = True,
        grounding_threshold: float = 0.6,
        prompts_path: Optional[Path] = None,
    ):
        """
        Initialize generator.

        Args:
            data_dir: Path to processed/wiki_extracted/
            llm: LLM for question generation
            validate_grounding: Whether to validate ground truth is in contexts
            grounding_threshold: Min ratio of key terms that must appear in contexts
            prompts_path: Path to YAML prompts file (default: generator/prompts/generation_prompt.yaml)
        """
        self.data_dir = Path(data_dir)
        self.llm = llm or LLMInference(provider="api")  # Use API provider for generation
        self.validate_grounding = validate_grounding
        self.grounding_threshold = grounding_threshold

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

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

            # Find all .jsonl files
            jsonl_files = list(folder_path.glob("*.jsonl"))

            if not jsonl_files:
                logger.warning(f"No .jsonl files in {folder_path}")
                continue

            for jsonl_file in jsonl_files:
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
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
                    logger.error(f"Failed to load {jsonl_file}: {e}")

        logger.info(f"Sampled {len(reservoir)} articles from {total_seen} total (memory-efficient)")
        return reservoir

    def generate_questions(
        self,
        num_questions: int = 1000,
        folders: Optional[List[str]] = None,
        distribution: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate test questions from Wikipedia articles.

        Args:
            num_questions: Number of questions to generate (default: 1000)
            folders: Folder names to sample from (default: AA-AJ)
            distribution: Question type distribution

        Returns:
            List of question dicts for RAGAS
        """
        if folders is None:
            # First 10 folders
            folders = [f"{chr(65)}{chr(65+i)}" for i in range(10)]  # AA, AB, ..., AJ

        if distribution is None:
            distribution = {
                "simple": 0.40,
                "comparison": 0.25,
                "multihop": 0.25,
                "temporal": 0.10,
            }

        # Calculate sample size: enough for generation with buffer (memory-efficient)
        # For 1000 questions with 50% buffer = 1500 attempts, we need ~500-1000 articles
        sample_size = max(num_questions, 1000)  # At least 1000 articles

        # Load articles using reservoir sampling (memory-efficient)
        articles = self.load_articles_sample(folders, sample_size=sample_size)

        if not articles:
            raise ValueError("No articles loaded")

        # Generate questions with retry loop to handle validation failures
        questions = []

        for qtype, ratio in distribution.items():
            target_count = int(num_questions * ratio)
            logger.info(f"Generating {target_count} {qtype} questions...")

            # Over-generate to account for validation failures (up to 50% extra attempts)
            attempts = 0
            max_attempts = int(target_count * 1.5)
            type_questions = []

            while len(type_questions) < target_count and attempts < max_attempts:
                if attempts > 0 and attempts % 10 == 0:
                    logger.info(f"  Progress: {len(type_questions)}/{target_count} (attempts: {attempts})")

                try:
                    question = self._generate_question(qtype, articles)
                    if question:
                        # Validate grounding if enabled
                        if self.validate_grounding:
                            if self._validate_grounding(question["ground_truth"], question["contexts"]):
                                type_questions.append(question)
                            else:
                                logger.debug(f"Skipping question - ground truth not grounded in contexts")
                        else:
                            type_questions.append(question)
                except Exception as e:
                    logger.warning(f"Failed to generate {qtype} question: {e}")

                attempts += 1

            logger.info(f"  Generated {len(type_questions)}/{target_count} {qtype} questions ({attempts} attempts)")
            questions.extend(type_questions)

        random.shuffle(questions)
        logger.info(f"Generated {len(questions)}/{num_questions} questions total (after validation)")

        if len(questions) < num_questions * 0.8:
            logger.warning(
                f"Generated significantly fewer questions than requested: "
                f"{len(questions)}/{num_questions} ({len(questions)/num_questions*100:.1f}%)"
            )

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
        text = article["text"][:1000]  # First 1000 chars

        # Load prompt from YAML
        prompt_template = self.prompts["generate_simple"]["template"]
        prompt = prompt_template.format(
            title=article['title'],
            text=text
        )

        response = self.llm.generate(
            prompt=prompt,
            temperature=0.7,
            max_new_tokens=300,
            chain_of_thought_enabled=False,
        )

        # Parse JSON
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
        text1 = article1["text"][:800]
        text2 = article2["text"][:800]

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
            temperature=0.7,
            max_new_tokens=400,
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
            logger.error(f"Failed to parse JSON: {e}")
            return None

    def _generate_multihop(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate multihop reasoning question from 2-3 articles."""

        num_articles = random.choice([2, 3])
        selected = random.sample(articles, num_articles)

        texts = [a["text"][:600] for a in selected]
        titles = [a["title"] for a in selected]

        articles_text = "\n\n".join([
            f"ARTICLE {i+1} TITLE: {titles[i]}\nTEXT:\n{texts[i]}"
            for i in range(num_articles)
        ])

        # Load prompt from YAML
        prompt_template = self.prompts["generate_multihop"]["template"]
        prompt = prompt_template.format(articles_text=articles_text)

        response = self.llm.generate(
            prompt=prompt,
            temperature=0.7,
            max_new_tokens=500,
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
            logger.error(f"Failed to parse JSON: {e}")
            return None

    def _generate_temporal(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate temporal/chronological question."""

        article = random.choice(articles)
        text = article["text"][:1000]

        # Load prompt from YAML
        prompt_template = self.prompts["generate_temporal"]["template"]
        prompt = prompt_template.format(
            title=article['title'],
            text=text
        )

        response = self.llm.generate(
            prompt=prompt,
            temperature=0.7,
            max_new_tokens=300,
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
            logger.error(f"Failed to parse JSON: {e}")
            return None

    def _extract_key_terms(self, text: str) -> Set[str]:
        """
        Extract key terms (words, numbers, dates) from text.
        Used for ground truth validation.
        """
        # Remove punctuation and convert to lowercase
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())

        # Split into tokens
        tokens = text_clean.split()

        # Filter stopwords (basic list)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'can', 'may', 'might', 'must', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those', 'what',
            'which', 'who', 'when', 'where', 'why', 'how', 'if', 'then', 'than',
        }

        # Keep tokens that are:
        # 1. Longer than 2 chars
        # 2. Not stopwords
        # 3. Or pure numbers/dates
        key_terms = set()
        for token in tokens:
            if len(token) > 2 and token not in stopwords:
                key_terms.add(token)
            elif re.fullmatch(r'\d+', token):  # Pure numbers/dates only
                key_terms.add(token)

        return key_terms

    def _validate_grounding(
        self,
        ground_truth: str,
        contexts: List[str],
    ) -> bool:
        """
        Validate that ground truth is grounded in contexts.

        Uses keyword matching: checks if key terms from ground truth
        appear in contexts at rate >= threshold.

        Args:
            ground_truth: Expected answer
            contexts: Context chunks

        Returns:
            True if ground truth is sufficiently grounded
        """
        # Extract key terms from ground truth
        gt_terms = self._extract_key_terms(ground_truth)

        if not gt_terms:
            # No key terms -> can't validate
            logger.warning(f"No key terms extracted from ground truth: {ground_truth[:50]}")
            return False

        # Combine all contexts
        contexts_text = " ".join(contexts).lower()

        # Count how many terms appear in contexts
        found_terms = 0
        for term in gt_terms:
            if term in contexts_text:
                found_terms += 1

        ratio = found_terms / len(gt_terms)

        if ratio < self.grounding_threshold:
            logger.debug(
                f"Ground truth not sufficiently grounded: "
                f"{found_terms}/{len(gt_terms)} terms found ({ratio:.2%} < {self.grounding_threshold:.2%})"
            )
            return False

        return True

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
