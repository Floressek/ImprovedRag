from __future__ import annotations
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
import shutil
import re

import requests
from tqdm import tqdm

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
        }


class WikiExtractor:
    """Extract Wikipedia articles using Attardi's WikiExtractor."""

    def __init__(
            self,
            min_text_length: int = 100,
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
                "uv pip install wikiextractor"
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
        if self.filter_disambig:
            cmd.append("--filter_disambig")
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
                logger.debug(f"WikiExtractor output: {result.stdout}")

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

    def extracted_from_json_dir(self, json_dir: Path) -> Iterator[WikiArticle]:
        """
        Load articles from a directory of JSON files.
        :param json_dir: Directory containing JSON files
        Yields:
            WikiArticle instances
        """
        if not json_dir.exists() or not json_dir.is_dir():
            raise FileNotFoundError(f"JSON directory not found: {json_dir}")

        yield from self._parse_extracted_files(json_dir)

    def _parse_extracted_files(self, output_dir: Path) -> Iterator[WikiArticle]:
        """
        Parse extracted files in the given directory.
        :param output_dir: Directory with extracted files
        Yields:
            WikiArticle instances
        """
        wiki_files = sorted(output_dir.glob("**/wiki_*.json"))

        if not wiki_files:
            logger.warning(f"No extracted files found in {output_dir}")
            return

        logger.info(f"Parsing {len(wiki_files)} extracted files from {output_dir}")

        for wiki_file in wiki_files:
            if self.max_articles and self._article_count >= self.max_articles:
                logger.info(f"Reached max articles limit: {self.max_articles}")
                break

            logger.debug(f"Processing file: {wiki_file}")

            try:
                if self.json_output:
                    yield from self._parse_json_file(wiki_file)
                else:
                    yield from self._parse_text_file(wiki_file)
            except Exception as e:
                logger.warning(f"Failed to parse {wiki_file}: {e}")
                continue

    def _parse_json_file(self, file_path: Path) -> Iterator[WikiArticle]:
        """
        Parse a JSON file and yield WikiArticle instances.
        :param file_path: Path to JSON file
        Yields:
            WikiArticle instances
        """
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    article = self._parse_json_article(data)

                    if article and len(article.text) >= self.min_text_length:
                        self._article_count += 1
                        yield article

                        if self._article_count % 1000 == 0:
                            logger.info(f"Extracted {self._article_count} articles so far")
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in {file_path}: {e}")
                    continue

    def _parse_json_article(self, data: dict) -> Optional[WikiArticle]:
        """
        Parse a single JSON article dictionary.
        :param data: Article data as dictionary
        Returns:
            WikiArticle instance or None if invalid
        """
        try:
            article_id = str(data.get("id", ""))
            title = data.get("title", "")
            text = data.get("text", "")
            url = data.get("url", "")

            if not article_id or not title or not text:
                return None

            text = self._clean_text(text)

            return WikiArticle(
                id=article_id,
                title=title,
                text=text,
                url=url,
                categories=[], # Categories not provided in JSON
            )
        except Exception as e:
            logger.warning(f"Error parsing article data: {e}")
            return None

    # FIXME: reduce the if levels to separate functions - 3 functions needed
    def _parse_text_file(self, file_path: Path) -> Iterator[WikiArticle]:
        """
        Parse a plain text file and yield WikiArticle instances.
        :param file_path: Path to text file
        Yields:
            WikiArticle instances
        """
        with open(file_path, "r", encoding="utf-8") as f:
            current_article = None
            current_text = []

            for line in f:
                # Article start
                if line.startswith('<doc id="'):
                    if current_article and current_text:
                        text = '\n'.join(current_text).strip()
                        if len(text) >= self.min_text_length:
                            current_article.text = self._clean_text(text)
                            self._article_count += 1
                            yield current_article

                    match = re.match(r'<doc id="([^"]+)" url="([^"]+)" title="([^"]+)">', line)
                    if match:
                        current_article = WikiArticle(
                            id=match.group(1),
                            title=match.group(3),
                            text="",
                            url=match.group(2),
                        )
                        current_text = []
                # End of article
                elif line.strip() == '</doc>':
                    if current_article and current_text:
                        text = '\n'.join(current_text).strip()
                        if len(text) >= self.min_text_length:
                            current_article.text = self._clean_text(text)
                            self._article_count += 1
                            yield current_article
                    current_article = None
                    current_text = []

                elif current_article is not None:
                    # Article content
                    current_text.append(line.rstrip())

    def _clean_text(self, text: str) -> str:
        """
        Clean article text by removing unwanted patterns.
        :param text: Raw article text
        Returns:
            Cleaned text
        """
        # WikiExtractor should handle most cleaning, but just in case:

        # Remove any remaining <ref> tags
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)

        # Remove excess whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        return text.strip()

# FIXME: prob shoould go to the controller folder for endpoint use
def download_wikipedia_dump(
        language: str = "en",
        dump_date: str = "latest",
        output_dir: Path = Path("data/raw"),
        dump_type: str = "pages-articles-multistream",
        chunk_number: Optional[int] = 1,
) -> Path:
    """
    Download Wikipedia dump file.

    Args:
        language: Wikipedia language code
        dump_date: Dump date or 'latest'
        output_dir: Output directory
        dump_type: Type of dump
        chunk_number: Chunk number for multistream (1-27 typically), None for full

    Returns:
        Path to downloaded file
    """
    base_url = f"https://dumps.wikimedia.org/{language}wiki/{dump_date}"

    if chunk_number:
        # Download smaller chunk for testing
        filename = f"{language}wiki-{dump_date}-{dump_type}{chunk_number}.xml.bz2"
    else:
        # Full dump
        filename = f"{language}wiki-{dump_date}-{dump_type}.xml.bz2"

    url = f"{base_url}/{filename}"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if output_path.exists():
        logger.info(f"Wikipedia dump already exists: {output_path}")
        return output_path

    logger.info(f"Downloading Wikipedia dump from {url} to {output_path}")
    logger.info("This may take a while depending on your internet connection...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=filename
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Downloaded dump to {output_path}")
        return output_path

    except requests.RequestException as e:
        logger.error(f"Failed to download Wikipedia dump: {e}")
        if output_path.exists():
            output_path.unlink() # remove incomplete file
        raise RuntimeError("Download failed") from e