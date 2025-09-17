from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

from .chunker import Chunk, TextChunker
from .wiki_extractor import WikiArticle, WikiExtractor

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the complete document ingestion process."""

    def __init__(
            self,
            extractor: Optional[WikiExtractor] = None,
            chunker: Optional[TextChunker] = None,
    ):
        """Initialize ingestion pipeline.

        Args:
            extractor: Wikipedia extractor instance
            chunker: Text chunker instance
        """
        self.extractor = extractor or WikiExtractor()
        self.chunker = chunker

    def ingest_wikipedia(
            self,
            source: str | Path,
            max_articles: Optional[int] = None,
            max_chunks: Optional[int] = None,
    ) -> Iterator[Chunk]:
        """Ingest Wikipedia data from dump or extracted JSON.

        Args:
            source: Path to Wikipedia dump or extracted JSON directory
            max_articles: Maximum articles to process
            max_chunks: Maximum chunks to generate

        Yields:
            Chunk objects
        """
        source = Path(source)

        # Configure extractor limits
        if max_articles:
            self.extractor.max_articles = max_articles

        # Determine source type and extract articles
        if source.is_file() and source.suffix in ['.xml', '.bz2']:
            logger.info(f"Processing Wikipedia dump: {source}")
            articles = self.extractor.extract_from_dump(source)
        elif source.is_dir():
            logger.info(f"Processing extracted Wikipedia JSON from: {source}")
            articles = self.extractor.extracted_from_json_dir(source)
        else:
            raise ValueError(f"Invalid source: {source}")

        # Process articles and generate chunks
        chunk_count = 0
        article_count = 0

        for article in articles:
            article_count += 1

            # Convert article to dict for chunking
            article_dict = article.to_dict()

            # Generate chunks if chunker is provided
            if self.chunker:
                chunks = self.chunker.chunk_document(
                    text=article.text,
                    doc_id=article.id,
                    doc_title=article.title,
                    metadata={
                        "url": article.url,
                        "categories": article.categories,
                    }
                )

                for chunk in chunks:
                    if max_chunks and chunk_count >= max_chunks:
                        logger.info(f"Reached max chunks limit: {max_chunks}")
                        return

                    chunk_count += 1
                    yield chunk
            else:
                # If no chunker, yield article as single chunk
                chunk = Chunk(
                    id=f"article_{article.id}",
                    text=article.text,
                    doc_id=article.id,
                    doc_title=article.title,
                    position=0,
                    total_chunks=1,
                    token_count=len(article.text.split()),
                    char_count=len(article.text),
                    metadata={
                        "url": article.url,
                        "categories": article.categories,
                    }
                )

                if max_chunks and chunk_count >= max_chunks:
                    logger.info(f"Reached max chunks limit: {max_chunks}")
                    return

                chunk_count += 1
                yield chunk

            if article_count % 100 == 0:
                logger.info(f"Processed {article_count} articles, {chunk_count} chunks")

    def ingest_documents(
            self,
            documents: Iterator[dict],
            text_field: str = "text",
            id_field: str = "id",
            title_field: str = "title",
            max_chunks: Optional[int] = None,
    ) -> Iterator[Chunk]:
        """Ingest generic documents.

        Args:
            documents: Iterator of document dictionaries
            text_field: Field containing document text
            id_field: Field containing document ID
            title_field: Field containing document title
            max_chunks: Maximum chunks to generate

        Yields:
            Chunk objects
        """
        if not self.chunker:
            raise ValueError("Chunker is required for document ingestion")

        chunk_count = 0

        for chunk in self.chunker.chunk_documents(
                documents,
                text_field=text_field,
                id_field=id_field,
                title_field=title_field,
        ):
            if max_chunks and chunk_count >= max_chunks:
                logger.info(f"Reached max chunks limit: {max_chunks}")
                return

            chunk_count += 1
            yield chunk

    def process_in_batches(
            self,
            chunks: Iterator[Chunk],
            batch_size: int = 100,
    ) -> Iterator[list[Chunk]]:
        """Process chunks in batches.

        Args:
            chunks: Iterator of chunks
            batch_size: Batch size

        Yields:
            Batches of chunks
        """
        batch = []

        for chunk in chunks:
            batch.append(chunk)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining batch
        if batch:
            yield batch