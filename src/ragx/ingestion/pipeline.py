from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from src.ragx.ingestion.chunkers.chunker import TextChunker
from src.ragx.ingestion.ingestion_pipeline import IngestionPipeline
from src.ragx.ingestion.wiki_extractor import WikiExtractor
from src.ragx.ingestion.utils.download_wiki_dump import download_wikipedia_dump
from src.ragx.utils.logging_config import setup_logging
from src.ragx.retrieval.embedder import Embedder
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """RAGx Wikipedia ingestion CLI."""
    setup_logging(level=settings.app.log_level)


@click.command()
@click.option("--language", default="en", help="Wikipedia language code (en, pl, etc.)")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None, help="Output directory")
@click.option("--dump-date", default="latest", help="Dump date (YYYYMMDD or 'latest')")
@click.option("--chunk-number", type=int, default=None, help="Multistream chunk number")
def download(language: str, output_dir: Optional[Path], dump_date: str, chunk_number: Optional[int]):
    """Download Wikipedia dump file."""
    output_dir = output_dir or Path(settings.app.raw_dir)
    click.echo(f"Downloading {language} Wikipedia dump...")
    try:
        dump_path = download_wikipedia_dump(
            language=language,
            dump_date=dump_date,
            output_dir=output_dir,
            chunk_number=chunk_number,
        )
        click.echo(f"✓ Downloaded to: {dump_path}")
    except Exception as e:
        click.echo(f"✗ Download failed: {e}", err=True)
        sys.exit(1)


@click.command()
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option("--max-articles", type=int, default=10000, help="Maximum articles to process")
@click.option("--max-chunks", type=int, default=None, help="Maximum chunks to generate")
@click.option("--recreate-collection", is_flag=True, help="Recreate Qdrant collection")
@click.option("--batch-size", type=int, default=100, help="Processing batch size")
@click.option("--embedding-model", default=None, help="Override embedding model")
@click.option("--chunk-size", type=int, default=None, help="Override chunk size")
@click.option("--chunk-overlap", type=int, default=None, help="Override chunk overlap")
def ingest(
        source: Path,
        max_articles: int,
        max_chunks: Optional[int],
        recreate_collection: bool,
        batch_size: int,
        embedding_model: Optional[str],
        chunk_size: Optional[int],
        chunk_overlap: Optional[int],
):
    """Ingest Wikipedia data into vector store.

    All configuration comes from .env unless explicitly overridden via CLI flags.
    """

    # Resolve overrides
    embedding_model = embedding_model or settings.embedder.model_id
    chunk_size = chunk_size or settings.chunker.chunk_size
    chunk_overlap = chunk_overlap or settings.chunker.chunk_overlap

    click.echo("Starting Wikipedia ingestion pipeline...")
    click.echo("Configuration:")
    click.echo(f"  Source: {source}")
    click.echo(f"  Embedding model: {embedding_model}")
    click.echo(f"  Chunk size: {chunk_size}")
    click.echo(f"  Chunk overlap: {chunk_overlap}")
    click.echo(f"  Qdrant: {settings.qdrant.url} / {settings.qdrant.collection_name}")

    try:
        # 1) Components
        click.echo("\n1. Initializing components...")

        # Wszystko z settings! Super proste!
        embedder = Embedder(model_id=embedding_model)
        click.echo(f"✓ Embedder initialized (dim={embedder.get_dimension()})")

        vector_store = QdrantStore(
            embedding_dim=embedder.get_dimension(),
            recreate_collection=recreate_collection,
        )
        click.echo("✓ Vector store ready")

        # 2) Chunker & pipeline (NO prefix in stored text — we add at embedding time)
        chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name_tokenizer=embedding_model,
            model_name_embedder=embedding_model,
            chunking_model_override=settings.chunker.chunking_model,
        )

        pipeline = IngestionPipeline(
            extractor=WikiExtractor(
                max_articles=max_articles,
                json_output=True,
                processes=6,
            ),
            chunker=chunker,
        )
        click.echo(f"✓ Pipeline created ({settings.chunker.strategy} chunking)")

        if settings.chunker.chunking_model:
            click.echo(f"  → Using lightweight model for boundaries: {settings.chunker.chunking_model}")

        # 3) Process & index
        click.echo("\n2. Processing Wikipedia articles...")

        chunks_iter = pipeline.ingest_wikipedia(
            source=source,
            max_articles=max_articles,
            max_chunks=max_chunks,
        )

        total_chunks = 0
        total_batches = 0

        for batch in pipeline.process_in_batches(chunks_iter, batch_size=batch_size):
            chunk_dicts = [c.to_dict() for c in batch]
            texts = [c["text"] for c in chunk_dicts]

            # Embed as passages (prefix applied here if enabled)
            embeddings = embedder.embed_texts(
                texts,
                show_progress=False,
                convert_to_numpy=True,
                add_prefix=True,
                prefix=None,  # will use passage_prefix when use_prefixes=True
            )
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()

            vector_store.add(
                vectors=embeddings,
                payloads=chunk_dicts,
                ids=[c["id"] for c in chunk_dicts],
            )

            total_chunks += len(batch)
            total_batches += 1
            if total_batches % 10 == 0:
                click.echo(f"  Processed {total_chunks} chunks in {total_batches} batches")

        click.echo("\n✓ Ingestion complete!")
        click.echo(f"  Total chunks: {total_chunks}")
        click.echo(f"  Total batches: {total_batches}")
        click.echo(f"  Points in collection: {vector_store.count()}")

    except Exception as e:
        click.echo(f"\n✗ Ingestion failed: {e}", err=True)
        logger.exception("Ingestion pipeline failed")
        sys.exit(1)


@cli.command()
def status():
    """Check vector store status."""
    try:
        vector_store = QdrantStore(recreate_collection=False)
        info = vector_store.get_collection_info()

        click.echo("Vector Store Status:")
        click.echo(f"  URL: {settings.qdrant.url}")
        click.echo(f"  Collection: {settings.qdrant.collection_name}")
        click.echo(f"  Points: {info['points_count']}")
        click.echo(f"  Vector size: {info['vector_size']}")
        click.echo(f"  Distance: {info['distance']}")
        click.echo(f"  Status: {info['status']}")
    except Exception as e:
        click.echo(f"✗ Failed to get status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query", type=str)
@click.option("--top-k", type=int, default=None, help="Number of results (default: from settings)")
@click.option("--embedding-model", default=None, help="Override embedding model")
def search(query: str, top_k: Optional[int], embedding_model: Optional[str]):
    """Search in the vector store."""
    top_k = top_k or settings.retrieval.context_top_n

    try:
        embedder = Embedder(model_id=embedding_model)  # None = uses settings
        vector_store = QdrantStore(
            embedding_dim=embedder.get_dimension(),
            recreate_collection=False,
        )

        qvec = embedder.embed_query(query)
        hits = vector_store.search(
            vector=qvec,
            top_k=top_k,
            hnsw_ef=settings.hnsw.search_ef,
        )

        click.echo(f"\nSearch results for: '{query}'\n")
        click.echo(f"Model: {embedder.model_id} | Top-K: {top_k} | HNSW EF: {settings.hnsw.search_ef}\n")

        for i, (pid, payload, score) in enumerate(hits, 1):
            click.echo(f"{i}. Score: {score:.4f}")
            click.echo(f"   Doc: {payload.get('doc_title', 'Unknown')}")
            click.echo(f"   Chunk: {payload.get('position', 0) + 1}/{payload.get('total_chunks', 0)}")
            text = payload.get("text", "")
            # payload text is clean (no 'passage:'), we embedded with prefix at encode time
            click.echo(f"   Text: {text[:200]}...")
            click.echo()

    except Exception as e:
        click.echo(f"✗ Search failed: {e}", err=True)
        logger.exception("Search failed")
        sys.exit(1)


# Register commands
cli.add_command(download)
cli.add_command(ingest)
cli.add_command(status)
cli.add_command(search)

if __name__ == "__main__":
    cli()
