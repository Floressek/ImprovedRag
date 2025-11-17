from __future__ import annotations

import logging
import json
import random
import re
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
    ):
        """
        Initialize generator.

        Args:
            data_dir: Path to processed/wiki_extracted/
            llm: LLM for question generation
            validate_grounding: Whether to validate ground truth is in contexts
            grounding_threshold: Min ratio of key terms that must appear in contexts
        """
        self.data_dir = Path(data_dir)
        self.llm = llm or LLMInference(provider="api")  # Use API provider for generation
        self.validate_grounding = validate_grounding
        self.grounding_threshold = grounding_threshold

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def load_articles(
        self,
        folders: List[str],
        max_articles_per_folder: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Load random articles from specified folders.

        Args:
            folders: List of folder names (e.g. ["AA", "AB", ...])
            max_articles_per_folder: Max articles to load per folder

        Returns:
            List of article dicts
        """
        articles = []

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

            logger.info(f"Loading from {folder} ({len(jsonl_files)} files)...")

            folder_articles = []

            for jsonl_file in jsonl_files:
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            article = json.loads(line.strip())

                            # Basic validation
                            if all(k in article for k in ["id", "title", "text", "url"]):
                                # Skip very short articles
                                if len(article["text"]) > 200:
                                    folder_articles.append(article)
                except Exception as e:
                    logger.error(f"Failed to load {jsonl_file}: {e}")

            # Random sample
            if len(folder_articles) > max_articles_per_folder:
                folder_articles = random.sample(folder_articles, max_articles_per_folder)

            articles.extend(folder_articles)
            logger.info(f"  Loaded {len(folder_articles)} articles from {folder}")

        logger.info(f"Total articles loaded: {len(articles)}")
        return articles

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

        # Load articles
        articles = self.load_articles(folders, max_articles_per_folder=200)

        if not articles:
            raise ValueError("No articles loaded")

        # Generate questions
        questions = []

        for qtype, ratio in distribution.items():
            count = int(num_questions * ratio)
            logger.info(f"Generating {count} {qtype} questions...")

            for i in range(count):
                if i % 10 == 0 and i > 0:
                    logger.info(f"  Progress: {i}/{count}")

                try:
                    question = self._generate_question(qtype, articles)
                    if question:
                        # Validate grounding if enabled
                        if self.validate_grounding:
                            if self._validate_grounding(question["ground_truth"], question["contexts"]):
                                questions.append(question)
                            else:
                                logger.debug(f"Skipping question - ground truth not grounded in contexts")
                        else:
                            questions.append(question)
                except Exception as e:
                    logger.warning(f"Failed to generate {qtype} question: {e}")

        random.shuffle(questions)
        logger.info(f"Generated {len(questions)} questions total (after validation)")

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

    def _generate_simple(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate simple factual question from 1 article."""

        article = random.choice(articles)
        text = article["text"][:1000]  # First 1000 chars

        prompt = f"""Based on this Wikipedia article, generate a SIMPLE factual question.

ARTICLE TITLE: {article['title']}

ARTICLE TEXT:
{text}

Generate a clear, answerable question that requires information from this article.

IMPORTANT:
- Question must be answerable from the text above
- ground_truth should be concise (1-3 sentences)
- RESPOND IN JSON ONLY

Response format (JSON):
{{
  "question": "What is...?",
  "ground_truth": "Concise answer based on the text"
}}
"""

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

        prompt = f"""Based on these two Wikipedia articles, generate a COMPARISON question.

ARTICLE 1 TITLE: {article1['title']}
ARTICLE 1 TEXT:
{text1}

ARTICLE 2 TITLE: {article2['title']}
ARTICLE 2 TEXT:
{text2}

Generate a question that compares or contrasts these two topics.
Use patterns like: "X vs Y", "Compare X and Y", "Difference between X and Y"

IMPORTANT:
- Question must be answerable from both texts
- ground_truth should compare/contrast both topics (2-4 sentences)
- RESPOND IN JSON ONLY

Response format (JSON):
{{
  "question": "Compare X and Y in terms of...?",
  "ground_truth": "X has... while Y has... The main difference is..."
}}
"""

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

        prompt = f"""Based on these Wikipedia articles, generate a MULTIHOP question.

{articles_text}

Generate a question that requires connecting information from MULTIPLE articles.
The question should need information from at least 2 different articles to answer fully.

Example patterns:
- "How did X influence Y, and what impact did that have on Z?"
- "What was X's contribution to Y?"
- "Connect A and B through their relationship with C"

IMPORTANT:
- Question must require reasoning across multiple articles
- ground_truth should synthesize info from all relevant articles (2-4 sentences)
- RESPOND IN JSON ONLY

Response format (JSON):
{{
  "question": "How did X relate to Y and influence Z?",
  "ground_truth": "X contributed to Y by... This influenced Z because..."
}}
"""

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

        prompt = f"""Based on this Wikipedia article, generate a TEMPORAL/CHRONOLOGICAL question.

ARTICLE TITLE: {article['title']}

ARTICLE TEXT:
{text}

Generate a question about time, sequence, dates, or chronology.
Examples: "When did X occur?", "What happened between X and Y?", "Timeline of X"

IMPORTANT:
- Question must involve time/dates/chronology
- ground_truth should mention specific dates/periods (1-3 sentences)
- RESPOND IN JSON ONLY

Response format (JSON):
{{
  "question": "When did X happen?",
  "ground_truth": "X occurred in YYYY when..."
}}
"""

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
        # 3. Or numbers/dates
        key_terms = set()
        for token in tokens:
            if len(token) > 2 and token not in stopwords:
                key_terms.add(token)
            elif re.match(r'\d+', token):  # Numbers/dates
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
