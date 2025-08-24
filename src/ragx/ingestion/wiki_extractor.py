from __future__ import annotations

import bz2
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
import shutil

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

        # Fixme: maybe switch?
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
