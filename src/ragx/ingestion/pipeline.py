from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from src.ragx.ingestion.chunker import TextChunker
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
    """RAGx Wikipedia ingestion CLI with optimized models."""
    setup_logging(level=settings.app.log_level)


@click.command()
@click.option(
    "--language",
    default="en",
    help="Wikipedia language code (en, pl, etc.)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=f"Directory to save Wikipedia dump (default: {settings.app.raw_dir})",
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
@click.option("--min-chunk-size", type=int, default=120, help="Minimum chunk size in tokens")
@click.option(
    "--max-chunk-size",
    type=int,
    default=None,
    help="Maximum chunk size in tokens (default: from settings)",
)
@click.option("--breakpoint-threshold", type=int, default=78, help="Semantic breakpoint percentile")
@click.option("--buffer-size", type=int, default=3, help="Buffer size for semantic chunking (2-5)")
@click.option(
    "--chunk-size",
    type=int,
    default=None,
    help="Chunk size in tokens (default: from settings)",
)
@click.option(
    "--chunk-overlap",
    type=int,
    default=None,
    help="Overlap between chunks (default: from settings)",
)
@click.option(
    "--embedding-model",
    default=None,
    help="Embedding model ID (default: from settings)",
)
@click.option(
    "--embedding-batch-size",
    type=int,
    default=None,
    help="Batch size for embedding (default: from settings)",
)
@click.option(
    "--use-prefixes/--no-prefixes",
    default=None,
    help="Use query:/passage: prefixes (default: from settings)",
)
@click.option(
    "--qdrant-url",
    default=None,
    help="Qdrant server URL (default: from settings)",
)
@click.option(
    "--collection-name",
    default=None,
    help="Qdrant collection name (default: from settings)",
)
@click.option("--recreate-collection", is_flag=True, help="Recreate collection if exists")
@click.option("--batch-size", type=int, default=100, help="Batch size for processing")
def ingest(
        source: Path,
        max_articles: int,
        max_chunks: Optional[int],
        min_chunk_size: int,
        max_chunk_size: Optional[int],
        buffer_size: int,
        breakpoint_threshold: int,
        chunk_size: Optional[int],
        chunk_overlap: Optional[int],
        embedding_model: Optional[str],
        embedding_batch_size: Optional[int],
        use_prefixes: Optional[bool],
        qdrant_url: Optional[str],
        collection_name: Optional[str],
        recreate_collection: bool,
        batch_size: int,
):
    """Ingest Wikipedia data into vector store with optimized settings."""

    embedding_model = embedding_model or settings.embedder.model_id
    embedding_batch_size = embedding_batch_size or settings.embedder.batch_size
    use_prefixes = use_prefixes if use_prefixes is not None else settings.embedder.use_prefixes
    chunk_size = chunk_size or settings.retrieval.chunk_size
    chunk_overlap = chunk_overlap or settings.retrieval.chunk_overlap
    max_chunk_size = max_chunk_size or settings.retrieval.chunk_size
    qdrant_url = qdrant_url or settings.qdrant.url
    collection_name = collection_name or settings.qdrant.collection_name

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
            batch_size=embedding_batch_size,
            use_prefixes=use_prefixes,
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
            strategy=settings.retrieval.chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            model_name_tokenizer=embedding_model,
            model_name_embedder=embedding_model,
            respect_sections=True,
            breakpoint_percentile_thresh=breakpoint_threshold,
            buffer_size=buffer_size,
            add_passage_prefix=False,
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

        click.echo("\n✓ Ingestion complete!")
        click.echo(f"  Total chunks: {total_chunks}")
        click.echo(f"  Total batches: {total_batches}")
        click.echo(f"  Points in collection: {vector_store.count()}")

    except Exception as e:
        click.echo(f"\n✗ Ingestion failed: {e}", err=True)
        logger.exception("Ingestion pipeline failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--qdrant-url",
    default=None,
    help=f"Qdrant server URL (default: {settings.qdrant.url})",
)
@click.option(
    "--collection-name",
    default=None,
    help=f"Qdrant collection name (default: {settings.qdrant.collection_name})",
)
def status(qdrant_url: Optional[str], collection_name: Optional[str]):
    """Check vector store status."""
    qdrant_url = qdrant_url or settings.qdrant.url
    collection_name = collection_name or settings.qdrant.collection_name
    try:
        vector_store = QdrantStore(
            url=qdrant_url,
            collection_name=collection_name,
            recreate_collection=False,
        )
        info = vector_store.get_collection_info()

        click.echo("Vector Store Status:")
        click.echo(f"  URL: {qdrant_url}")
        click.echo(f"  Collection: {collection_name}")
        click.echo(f"  Points: {info['points_count']}")
        click.echo(f"  Vector size: {info['vector_size']}")
        click.echo(f"  Distance: {info['distance']}")
        click.echo(f"  Status: {info['status']}")
    except Exception as e:
        click.echo(f"✗ Failed to get status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query", type=str)
@click.option("--top-k", type=int, default=None,
              help=f"Number of results to return (default: {settings.retrieval.context_top_n})")
@click.option("--embedding-model", default=None,
              help=f"Embedding model ID (default: {settings.embedder.model_id})")
@click.option("--use-prefixes/--no-prefixes", default=None,
              help=f"Use query:/passage: prefixes (default: {settings.embedder.use_prefixes})")
@click.option("--trust-remote-code/--no-trust-remote-code", default=True)
@click.option("--qdrant-url", default=None, help=f"Qdrant server URL (default: {settings.qdrant.url})")
@click.option("--collection-name", default=None,
              help=f"Qdrant collection name (default: {settings.qdrant.collection_name})")
def search(
        query: str,
        top_k: Optional[int],
        embedding_model: Optional[str],
        use_prefixes: Optional[bool],
        trust_remote_code: bool,
        qdrant_url: Optional[str],
        collection_name: Optional[str],
):
    """Test search functionality with optimized models."""
    top_k = top_k or settings.retrieval.context_top_n
    qdrant_url = qdrant_url or settings.qdrant.url
    collection_name = collection_name or settings.qdrant.collection_name
    try:
        embedder = Embedder(
            model_id=embedding_model,
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
        hits = vector_store.search(
            vector=qvec,
            top_k=top_k,
            hnsw_ef=settings.hnsw.search_ef,
        )

        click.echo(f"\nSearch results for: '{query}'\n")
        click.echo(f"Configuration:")
        click.echo(f"  Model: {embedder.model_id}")
        click.echo(f"  Use prefixes: {embedder.use_prefixes}")
        click.echo(f"  Top-K: {top_k}")
        click.echo(f"  HNSW EF: {settings.hnsw.search_ef}\n")

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
