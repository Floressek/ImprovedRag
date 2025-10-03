from __future__ import annotations
import logging

import os
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from src.ragx.ingestion.chunker import TextChunker
from src.ragx.ingestion.ingestion_pipeline import IngestionPipeline
from src.ragx.ingestion.wiki_extractor import WikiExtractor
from src.ragx.ingestion.utils.download_wiki_dump import download_wikipedia_dump
from src.ragx.logging_config import setup_logging
from src.ragx.retrieval.embedder import Embedder
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
load_dotenv()

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """RAGx Wikipedia ingestion CLI with optimized models."""
    setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))


@click.command()
@click.option(
    "--language",
    default="en",
    help="Wikipedia language code (en, pl, etc.)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/raw"),
    help="Directory to save Wikipedia dump",
)
@click.option(
    "--dump-date",
    default="latest",
    help="Wikipedia dump date (YYYYMMDD or 'latest')",
)
@click.option(
    "--chunk-number",
    type=int,
    default=None,
    help="Chunk number for multistream dumps (None = full file)",
)
def download(language: str, output_dir: Path, dump_date: str, chunk_number: Optional[int]):
    """Download Wikipedia dump file."""
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
@click.option("--chunk-size", type=int, default=512, help="Chunk size in tokens")
@click.option("--chunk-overlap", type=int, default=96, help="Overlap between chunks")
@click.option(
    "--embedding-model",
    default="Alibaba-NLP/gte-multilingual-base",
    help="Embedding model ID",
)
@click.option("--embedding-batch-size", type=int, default=64, help="Batch size for embedding")
@click.option(
    "--use-prefixes/--no-prefixes",
    default=True,
    help="Use query:/passage: prefixes for E5/GTE models",
)
@click.option("--qdrant-url", default=None, help="Qdrant server URL (env QDRANT_URL if not set)")
@click.option("--collection-name", default=None, help="Qdrant collection name (env QDRANT_COLLECTION)")
@click.option("--recreate-collection", is_flag=True, help="Recreate collection if exists")
@click.option("--batch-size", type=int, default=100, help="Batch size for processing")
def ingest(
        source: Path,
        max_articles: int,
        max_chunks: Optional[int],
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        embedding_batch_size: int,
        use_prefixes: bool,
        qdrant_url: Optional[str],
        collection_name: Optional[str],
        recreate_collection: bool,
        batch_size: int,
):
    """Ingest Wikipedia data into vector store with optimized settings."""

    # Get configuration from env if not provided
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "ragx_documents_v2")

    click.echo("Starting Wikipedia ingestion pipeline...")
    click.echo("Configuration:")
    click.echo(f"  Source: {source}")
    click.echo(f"  Embedding model: {embedding_model}")
    click.echo(f"  Embedding batch size: {embedding_batch_size}")
    click.echo(f"  Use prefixes: {use_prefixes}")
    click.echo(f"  Chunk size: {chunk_size}")
    click.echo(f"  Chunk overlap: {chunk_overlap}")
    click.echo(f"  Qdrant URL: {qdrant_url}")
    click.echo(f"  Collection: {collection_name}")

    try:
        # 1) Components
        click.echo("\n1. Initializing components...")

        embedder = Embedder(
            model_id=embedding_model,
            device="auto",
            normalize_embeddings=True,
            batch_size=embedding_batch_size,
            show_progress=True,
            max_seq_length=512,
            use_prefixes=use_prefixes,
            query_prefix="query: ",
            passage_prefix="passage: ",
        )
        click.echo(f"✓ Embedder initialized (dim={embedder.get_dimension()})")

        vector_store = QdrantStore(
            url=qdrant_url,
            collection_name=collection_name,
            embedding_dim=embedder.get_dimension(),
            recreate_collection=recreate_collection,
        )
        click.echo("✓ Vector store ready")

        # 2) Chunker & pipeline (NO prefix in stored text — we add at embedding time)
        chunker = TextChunker(
            strategy="semantic",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=100,
            model_name_tokenizer=embedding_model,
            model_name_embedder=embedding_model,
            respect_sections=True,
            add_passage_prefix=False,  # keep payload clean; add prefix only when embedding
        )

        pipeline = IngestionPipeline(
            extractor=WikiExtractor(
                max_articles=max_articles,
                json_output=True,
                processes=4,
            ),
            chunker=chunker,
        )
        click.echo("✓ Ingestion pipeline created with semantic chunking")

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

        click.echo(f"\n✓ Ingestion complete!")
        click.echo(f"  Total chunks: {total_chunks}")
        click.echo(f"  Total batches: {total_batches}")
        click.echo(f"  Points in collection: {vector_store.count()}")

    except Exception as e:
        click.echo(f"\n✗ Ingestion failed: {e}", err=True)
        logger.exception("Ingestion pipeline failed")
        sys.exit(1)


@cli.command()
@click.option("--qdrant-url", default=None, help="Qdrant server URL")
@click.option("--collection-name", default=None, help="Qdrant collection name")
def status(qdrant_url: Optional[str], collection_name: Optional[str]):
    """Check vector store status."""
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "ragx_documents_v2")
    try:
        vector_store = QdrantStore(
            url=qdrant_url,
            collection_name=collection_name,
            recreate_collection=False,
        )
        click.echo("Vector Store Status:")
        click.echo(f"  URL: {qdrant_url}")
        click.echo(f"  Collection: {collection_name}")
        click.echo(f"  Points: {vector_store.count()}")
    except Exception as e:
        click.echo(f"✗ Failed to get status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query", type=str)
@click.option("--top-k", type=int, default=5, help="Number of results to return")
@click.option("--embedding-model", default="Alibaba-NLP/gte-multilingual-base", help="Embedding model ID")
@click.option("--use-prefixes/--no-prefixes", default=True, help="Use query:/passage: prefixes")
@click.option("--trust-remote-code/--no-trust-remote-code", default=True)
@click.option("--qdrant-url", default=None, help="Qdrant server URL")
@click.option("--collection-name", default=None, help="Qdrant collection name")
def search(
        query: str,
        top_k: int,
        embedding_model: str,
        use_prefixes: bool,
        trust_remote_code: bool,
        qdrant_url: Optional[str],
        collection_name: Optional[str],
):
    """Test search functionality with optimized models."""
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "ragx_documents_v2")
    try:
        embedder = Embedder(
            model_id=embedding_model,
            device="auto",
            normalize_embeddings=True,
            use_prefixes=use_prefixes,
            trust_remote_code=trust_remote_code,
        )
        vector_store = QdrantStore(
            url=qdrant_url,
            collection_name=collection_name,
            embedding_dim=embedder.get_dimension(),
            recreate_collection=False,
        )

        qvec = embedder.embed_query(query)
        hits = vector_store.search(vector=qvec, top_k=top_k)

        click.echo(f"\nSearch results for: '{query}'\n")
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
        sys.exit(1)


# Register commands
cli.add_command(download)
cli.add_command(ingest)
cli.add_command(status)
cli.add_command(search)

if __name__ == "__main__":
    cli()
