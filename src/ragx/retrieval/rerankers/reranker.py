from __future__ import annotations

import logging
from typing import Any, Optional, Union

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-Encoder based reranker for improving retrieval precision."""

    def __init__(
            self,
            model_id: str = "jinaai/jina-reranker-v2-base-multilingual",
            device: Optional[str] = None,
            batch_size: int = 16,
            max_length: int = 512,
            show_progress: bool = False,
    ):
        """Initialize reranker.

        Args:
            model_id: HuggingFace model ID for cross-encoder
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for reranking
            max_length: Maximum sequence length
            show_progress: Whether to show progress bar
            cache_dir: Cache directory for model files
        """
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.show_progress = show_progress

        # Determine device
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading reranker model: {model_id} on {self.device}")

        # Load cross-encoder model
        self.model = CrossEncoder(
            model_id,
            device=self.device,
            max_length=max_length,
        )

        logger.info(f"Reranker initialized: {model_id}")

    def rerank(
            self,
            query: str,
            documents: list[dict[str, Any]],
            top_k: Optional[int] = None,
            text_field: str = "text",
            batch_size: Optional[int] = None,
            return_scores: bool = True,
    ) -> list[tuple[dict[str, Any], float]]:
        """Rerank documents based on relevance to query.

        Args:
            query: Query text
            documents: List of document dictionaries
            top_k: Number of top documents to return (None = all)
            text_field: Field in document dict containing text
            batch_size: Override default batch size
            return_scores: Whether to return scores with documents

        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not documents:
            return []

        batch_size = batch_size or self.batch_size

        # Extract texts and create pairs
        texts = []
        valid_docs = []

        for doc in documents:
            text = doc.get(text_field, "")
            if text:
                texts.append(text)
                valid_docs.append(doc)

        if not texts:
            return []

        # Create query-document pairs
        pairs = [[query, text] for text in texts]

        # Score pairs
        logger.debug(f"Reranking {len(pairs)} documents")

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=self.show_progress,
        )

        # Sort by score (higher is better)
        doc_scores = list(zip(valid_docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k if specified
        if top_k is not None and top_k < len(doc_scores):
            doc_scores = doc_scores[:top_k]

        if return_scores:
            return doc_scores
        else:
            return [doc for doc, _ in doc_scores]

    def rerank_passages(
            self,
            query: str,
            passages: list[str],
            top_k: Optional[int] = None,
            batch_size: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """Rerank raw text passages.

        Args:
            query: Query text
            passages: List of passage texts
            top_k: Number of top passages to return
            batch_size: Override default batch size

        Returns:
            List of (passage_index, score) tuples sorted by relevance
        """
        if not passages:
            return []

        batch_size = batch_size or self.batch_size

        # Create pairs
        pairs = [[query, passage] for passage in passages]

        # Score pairs
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=self.show_progress,
        )

        # Sort by score with original indices
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k if specified
        if top_k is not None and top_k < len(indexed_scores):
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores

    def batch_rerank(
            self,
            queries: list[str],
            documents_list: list[list[dict[str, Any]]],
            top_k: Optional[int] = None,
            text_field: str = "text",
    ) -> list[list[tuple[dict[str, Any], float]]]:
        """Rerank multiple queries with their respective documents.

        Args:
            queries: List of query texts
            documents_list: List of document lists (one per query)
            top_k: Number of top documents per query
            text_field: Field in document dict containing text

        Returns:
            List of reranked results for each query
        """
        if len(queries) != len(documents_list):
            raise ValueError("Number of queries must match number of document lists")

        results = []

        for query, documents in zip(queries, documents_list):
            reranked = self.rerank(
                query=query,
                documents=documents,
                top_k=top_k,
                text_field=text_field,
            )
            results.append(reranked)

        return results

    def warmup(self) -> None:
        """Warmup the model with a dummy input."""
        dummy_pairs = [["query", "document"]]
        _ = self.model.predict(dummy_pairs)
        logger.info("Reranker warmed up")