from __future__ import annotations
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Callable
import shutil
import re

logger = logging.getLogger(__name__)


@dataclass
class WikiArticle:
    """
    Represents a single Wikipedia article.
    """
    id: str
    title: str
    text: str
    url: Optional[str] = None
    categories: list[str] = field(default_factory=list)
    source_file: Optional[str] = None

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization
        """
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "url": self.url or f"https://en.wikipedia.org/wiki/{self.title.replace(' ', '_')}",
            "categories": self.categories,
            "source_file": self.source_file,
        }


class WikiExtractor:
    """Extract Wikipedia articles using Attardi's WikiExtractor."""

    def __init__(
            self,
            min_text_length: int = 10,
            max_articles: Optional[int] = None,
            processes: int = 4,
            bytes_per_file: str = "1M",
            quiet: bool = False,
            json_output: bool = True,
            no_templates: bool = True,
            filter_disambig: bool = True,
    ):
        """
        Args:
            min_text_length: Minimum text length to keep article
            max_articles: Maximum articles to extract
            processes: Number of parallel processes
            bytes_per_file: Size of each output file (e.g., "1M", "500K")
            quiet: Suppress WikiExtractor output
            json_output: Output in JSON format
            no_templates: Skip template expansion
            filter_disambig: Filter disambiguation pages
        """
        self.min_text_length = min_text_length
        self.max_articles = max_articles
        self.processes = processes
        self.bytes_per_file = bytes_per_file
        self.quiet = quiet
        self.json_output = json_output
        self.no_templates = no_templates
        self.filter_disambig = filter_disambig
        self._article_count = 0

        # Check whether WikiExtractor is installed
        self._check_wikiextractor_installed()

    def _check_wikiextractor_installed(self) -> None:
        """Check if WikiExtractor is installed, else raise error."""
        try:
            import wikiextractor
            version = getattr(wikiextractor, '__version__', 'unknown')
            logger.info(f"WikiExtractor version {version} is installed.")
        except ImportError:
            logger.error(
                "WikiExtractor not installed. Install with:\n"
                "pip install git+https://github.com/attardi/wikiextractor.git@ab8988ebfa9e4557411f3d4c0f4ccda139e18875"
            )
            raise ImportError("WikiExtractor required and not installed.")

    def extract_from_dump(
            self,
            dump_path: Path,
            output_dir: Optional[Path] = None,
            keep_extracted: bool = False,
    ) -> Iterator[WikiArticle]:
        """
        Extract articles from Wikipedia dump.
        :param dump_path: Path to Wikipedia dump (bz2 file)
        :param output_dir: Directory for extracted files (temp if None)
        :param keep_extracted: Keep extracted files after processing

        Yields:
            WikiArticle instances
        """
        if not dump_path.exists():
            raise FileNotFoundError(f"Wikipedia dump not found: {dump_path}")

        if output_dir is None:
            output_dir = Path("data/processed/wiki_extracted")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting wikipedia dump: {dump_path}")
        logger.info(f"Using output directory: {output_dir}")

        # Run WikiExtractor as a subprocess
        cmd = [
            sys.executable, "-m", "wikiextractor.WikiExtractor",
            str(dump_path),
            "--output", str(output_dir),
            "--bytes", self.bytes_per_file,
            "--processes", str(self.processes),
        ]

        if self.json_output:
            cmd.append("--json")
        if self.no_templates:
            cmd.append("--no-templates")
        if self.quiet:
            cmd.append("--quiet")

        if not self.quiet:
            cmd.append("--debug")

        logger.debug(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            if result.stdout and not self.quiet:
                logger.debug("WikiExtractor output: %s", result.stdout)

            # Parse the extracted files
            yield from self._parse_extracted_files(output_dir)

        except subprocess.CalledProcessError as e:
            logger.error(f"WikiExtractor failed: {e.stderr}")
            raise RuntimeError("WikiExtractor execution failed") from e
        finally:
            # For cleanup purposes
            if not keep_extracted and output_dir.exists():
                logger.info(f"Cleaning up extracted files in {output_dir}")
                shutil.rmtree(output_dir)

    def extracted_from_json_dir(
            self,
            json_dir: Path,
            skip_files: Optional[set[str]] = None,
            start_from_file: Optional[str] = None,
            progress_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Iterator[WikiArticle]:
        """
        Load articles from a directory of JSON files.

        Args:
            json_dir: Directory containing JSON files
            skip_files: Set of file paths to skip (already processed)
            start_from_file: Start processing from this file (inclusive)
            progress_callback: Callback(event, file_path) called on file start/complete

        Yields:
            WikiArticle instances
        """
        if not json_dir.exists() or not json_dir.is_dir():
            raise FileNotFoundError(f"JSON directory not found: {json_dir}")

        skip_files = skip_files or set()

        wiki_files = sorted(json_dir.glob("**/wiki_*"))
        wiki_files = [f for f in wiki_files if f.is_file() and not f.suffix]

        if not wiki_files:
            logger.warning(f"No extracted files found in {json_dir}")
            return

        logger.debug(f"Found {len(wiki_files)} extracted files in {json_dir}")

        if start_from_file:
            try:
                start_idx = next(i for i, f in enumerate(wiki_files) if f.name == start_from_file or str(f) == start_from_file)
                wiki_files = wiki_files[start_idx:]
                logger.info(f"Starting from file: {start_from_file} (skipping {start_idx} files)")
            except StopIteration:
                logger.warning(f"Start file {start_from_file} not found. Processing all files.")

        for wiki_file in wiki_files:
            file_str = str(wiki_file)

            if file_str in skip_files:
                logger.info(f"Skipping already processed file: {wiki_file.name}")
                continue

            if self.max_articles and self._article_count >= self.max_articles:
                logger.info(f"Reached max articles limit: {self.max_articles}")
                break

            logger.info(f"Processing file: {wiki_file.name}")

            # Notify start
            if progress_callback:
                progress_callback("start", file_str)

            try:
                file_article_count = 0
                with open(wiki_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            doc = json.loads(line)

                            # Required fields
                            doc_id = doc.get('id')
                            url = doc.get('url')
                            title = doc.get('title')
                            text = doc.get('text', '').strip()

                            # Omit short articles
                            if not text or len(text) < self.min_text_length:
                                continue

                            text = self._clean_text(text)

                            self._article_count += 1
                            file_article_count += 1

                            yield WikiArticle(
                                id=doc_id,
                                url=url,
                                title=title,
                                text=text,
                                source_file=file_str,
                            )

                            if self.max_articles and self._article_count >= self.max_articles:
                                logger.info(f"Reached max articles limit: {self.max_articles}")
                                break

                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Invalid JSON at {wiki_file}:{line_num} - {e}"
                            )
                            continue

                logger.info(f"Completed {wiki_file.name}: {file_article_count} articles")
                if progress_callback:
                    progress_callback("complete", file_str)

            except Exception as e:
                logger.error(f"Error reading {wiki_file}: {e}")
                continue

    def reset_counter(self) -> None:
        """Reset article counter."""
        self._article_count = 0

    def _parse_extracted_files(self, output_dir: Path) -> Iterator[WikiArticle]:
        """
        Parse extracted files in JSONL format (one JSON object per line).
        WikiExtractor with --json flag produces files named wiki_XX (no extension).

        :param output_dir: Directory with extracted files
        Yields:
            WikiArticle instances
        """
        yield from self.extracted_from_json_dir(output_dir)

    def _clean_text(self, text: str) -> str:
        """
        Clean article text by removing unwanted patterns.
        :param text: Raw article text
        Returns:
            Cleaned text
        """
        # Remove any remaining <ref> tags
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)

        # Remove excess whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        return text.strip()

    # def _parse_json_file(self, file_path: Path) -> Iterator[WikiArticle]:
    #     """
    #     Parse a JSON file and yield WikiArticle instances.
    #     :param file_path: Path to JSON file
    #     Yields:
    #         WikiArticle instances
    #     """
    #     with open(file_path, "r", encoding="utf-8") as f:
    #         for line in f:
    #             if not line.strip():
    #                 continue
    #             try:
    #                 data = json.loads(line)
    #                 article = self._parse_json_article(data)
    #
    #                 if article and len(article.text) >= self.min_text_length:
    #                     self._article_count += 1
    #                     yield article
    #
    #                     if self._article_count % 1000 == 0:
    #                         logger.info(f"Extracted {self._article_count} articles so far")
    #             except json.JSONDecodeError as e:
    #                 logger.warning(f"JSON decode error in {file_path}: {e}")
    #                 continue
    #
    # def _parse_json_article(self, data: dict) -> Optional[WikiArticle]:
    #     """
    #     Parse a single JSON article dictionary.
    #     :param data: Article data as dictionary
    #     Returns:
    #         WikiArticle instance or None if invalid
    #     """
    #     try:
    #         article_id = str(data.get("id", ""))
    #         title = data.get("title", "")
    #         text = data.get("text", "")
    #         url = data.get("url", "")
    #
    #         if not article_id or not title or not text:
    #             return None
    #
    #         text = self._clean_text(text)
    #
    #         return WikiArticle(
    #             id=article_id,
    #             title=title,
    #             text=text,
    #             url=url,
    #             categories=[],  # Categories not provided in JSON
    #         )
    #     except Exception as e:
    #         logger.warning(f"Error parsing article data: {e}")
    #         return None
    #
    # # FIXME: reduce the if levels to separate functions - 3 functions needed
    # def _parse_text_file(self, file_path: Path) -> Iterator[WikiArticle]:
    #     """
    #     Parse a plain text file and yield WikiArticle instances.
    #     :param file_path: Path to text file
    #     Yields:
    #         WikiArticle instances
    #     """
    #     with open(file_path, "r", encoding="utf-8") as f:
    #         current_article = None
    #         current_text = []
    #
    #         for line in f:
    #             # Article start
    #             if line.startswith('<doc id="'):
    #                 if current_article and current_text:
    #                     text = '\n'.join(current_text).strip()
    #                     if len(text) >= self.min_text_length:
    #                         current_article.text = self._clean_text(text)
    #                         self._article_count += 1
    #                         yield current_article
    #
    #                 match = re.match(r'<doc id="([^"]+)" url="([^"]+)" title="([^"]+)">', line)
    #                 if match:
    #                     current_article = WikiArticle(
    #                         id=match.group(1),
    #                         title=match.group(3),
    #                         text="",
    #                         url=match.group(2),
    #                     )
    #                     current_text = []
    #             # End of article
    #             elif line.strip() == '</doc>':
    #                 if current_article and current_text:
    #                     text = '\n'.join(current_text).strip()
    #                     if len(text) >= self.min_text_length:
    #                         current_article.text = self._clean_text(text)
    #                         self._article_count += 1
    #                         yield current_article
    #                 current_article = None
    #                 current_text = []
    #
    #             elif current_article is not None:
    #                 # Article content
    #                 current_text.append(line.rstrip())
