from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
            output_path.unlink()  # remove incomplete file
        raise RuntimeError("Download failed") from e
