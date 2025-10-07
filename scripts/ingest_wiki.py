from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.ragx.ingestion.chunkers.chunker import TextChunker
from src.ragx.ingestion.ingestion_pipeline import IngestionPipeline
from src.ragx.ingestion.wiki_extractor import WikiExtractor
from src.ragx.ingestion.utils.download_wiki_dump import download_wikipedia_dump
from src.ragx.utils.logging_config import setup_logging
from src.ragx.retrieval.embedder import Embedder
from src.ragx.retrieval.vector_stores.qdrant_store import QdrantStore


load_dotenv()
logger = logging.getLogger(__name__)


def main() -> None:
    """Main ingestion script."""
    parser = argparse.ArgumentParser(description="Ingest Wikipedia data into RAGx")

    parser.add_argument(
        "--source",
        type=Path,
        help="Path to Wikipedia dump or extracted JSON directory",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download Wikipedia dump first",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Wikipedia language code (en, pl, etc.)",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=10000,
        help="Maximum articles to process",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        help="Maximum chunks to generate",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in tokens",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=96,
        help="Overlap between chunks",
    )
    parser.add_argument(
        "--chunking-strategy",
        choices=["semantic", "token"],
        default="semantic",
        help="Chunking strategy",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=120,
        help="Minimum chunk size in tokens",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=480,
        help="Maximum chunk size in tokens",
    )
    parser.add_argument(
        "--breakpoint-threshold",
        type=int,
        default=78,
        help="Semantic breakpoint percentile (75-90)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=3,
        help="Buffer size for semantic chunking",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "Alibaba-NLP/gte-multilingual-base"),
        help="Embedding model ID",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Batch size for embedding",
    )
    parser.add_argument(
        "--use-prefixes",
        action="store_true",
        default=True,
        help="Use query:/passage: prefixes for E5/GTE models",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant server URL",
    )
    parser.add_argument(
        "--collection-name",
        default=os.getenv("QDRANT_COLLECTION", "ragx_documents_v2"),
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Recreate collection if exists",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--chunk-number",
        type=int,
        default=1,
        help="Wikipedia dump chunk number (for multistream dumps)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    logger.info("Starting Wikipedia ingestion pipeline")
    logger.info("Configuration:")
    logger.info("  Language: %s", args.language)
    logger.info("  Max articles: %d", args.max_articles)
    logger.info("  Chunk size: %d", args.chunk_size)
    logger.info("  Chunk overlap: %d", args.chunk_overlap)
    logger.info("  Chunking strategy: %s", args.chunking_strategy)
    logger.info("  Embedding model: %s", args.embedding_model)
    logger.info("  Use prefixes: %s", args.use_prefixes)
    logger.info("  Qdrant URL: %s", args.qdrant_url)
    logger.info("  Collection: %s", args.collection_name)

    try:
        # Step 1: Handle source data
        if args.download:
            logger.info(f"Downloading {args.language} Wikipedia dump...")
            source_path = download_wikipedia_dump(
                language=args.language,
                dump_date="latest",
                output_dir=Path("data/raw"),
                chunk_number=args.chunk_number,
            )
        elif args.source:
            source_path = args.source
            if not source_path.exists():
                raise FileNotFoundError(f"Source not found: {source_path}")
        else:
            # Default: look for existing data
            source_path = Path("data/processed/wiki_extracted")
            if not source_path.exists():
                source_path = Path("data/raw/pl_wiki_dump")
                # Try to find any wiki dump
                wiki_files = list(source_path.glob("*wiki*.bz2"))
                if wiki_files:
                    source_path = wiki_files[0]
                else:
                    raise FileNotFoundError(
                        "No source specified. Use --source or --download"
                    )

        logger.info(f"Using source: {source_path}")

        # Step 2: Initialize components
        logger.info("Initializing components...")

        # Create embedder
        embedder = Embedder(
            model_id=args.embedding_model,
            device="auto",
            normalize_embeddings=True,
            batch_size=args.embedding_batch_size,
            show_progress=True,
            max_seq_length=512,
            use_prefixes=args.use_prefixes,
            query_prefix="query: ",
            passage_prefix="passage: ",
        )
        logger.info(f"Embedder initialized (dim={embedder.get_dimension()})")

        # Create vector store
        vector_store = QdrantStore(
            url=args.qdrant_url,
            collection_name=args.collection_name,
            embedding_dim=embedder.get_dimension(),
            recreate_collection=args.recreate_collection,
        )

        collection_info = vector_store.get_collection_info()
        logger.info(f"Vector store ready (current points={collection_info['points_count']})")

        # Create chunker
        chunker = TextChunker(
            strategy=args.chunking_strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_chunk_size=args.min_chunk_size,
            max_chunk_size=args.max_chunk_size,
            model_name_tokenizer=args.embedding_model,
            model_name_embedder=args.embedding_model,
            respect_sections=True,
            breakpoint_percentile_thresh=args.breakpoint_threshold,
            buffer_size=args.buffer_size,
            add_passage_prefix=False,
        )

        # Create ingestion pipeline
        pipeline = IngestionPipeline(
            extractor=WikiExtractor(
                max_articles=args.max_articles,
                json_output=True,
                processes=4,
            ),
            chunker=chunker,
        )

        # Step 3: Process and index data
        logger.info("Processing Wikipedia articles...")

        chunks_iterator = pipeline.ingest_wikipedia(
            source=source_path,
            max_articles=args.max_articles,
            max_chunks=args.max_chunks,
        )

        # Process in batches
        total_chunks = 0
        total_batches = 0

        for batch in pipeline.process_in_batches(chunks_iterator, batch_size=args.batch_size):
            # Convert chunks to dictionaries
            chunk_dicts = [chunk.to_dict() for chunk in batch]

            # Extract texts for embedding
            texts = [chunk["text"] for chunk in chunk_dicts]

            # "passage" prefix will come from the embedder if enabled
            embeddings = embedder.embed_texts(
                texts,
                show_progress=False,
                convert_to_numpy=False,
            )

            # Add to vector store
            vector_store.add(
                vectors=embeddings,
                payloads=chunk_dicts,
                ids=[chunk["id"] for chunk in chunk_dicts],
            )

            total_chunks += len(batch)
            total_batches += 1

            if total_batches % 10 == 0:
                logger.info(f"Processed {total_chunks} chunks in {total_batches} batches")

        # Final statistics
        logger.info("=" * 50)
        logger.info("Ingestion complete!")
        logger.info(f"  Total chunks processed: {total_chunks}")
        logger.info(f"  Total batches: {total_batches}")

        final_info = vector_store.get_collection_info()
        logger.info(f"  Total points in collection: {final_info['points_count']}")
        logger.info("=" * 50)

        # Test search
        if total_chunks > 0:
            logger.info("\nTesting search functionality...")
            test_query = "artificial intelligence"

            # Embed query with prefix if needed
            query_vector = embedder.embed_query(test_query)
            results = vector_store.search(query_vector, top_k=10)

            logger.info(f"Test search for '{test_query}':")
            for i, (id_, payload, score) in enumerate(results, 1):
                logger.info(f"  {i}. Score: {score:.4f} - {payload.get('doc_title', 'Unknown')}")

    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()