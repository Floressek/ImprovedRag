from __future__ import annotations

import logging
from typing import Optional, Union, Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Embeds text using a specified model from SentenceTransformers."""

    def __init__(
            self,
            model_id: str = "thenlper/gte-multilingual-base",
            device: Optional[str] = None,
            normalize_embeddings: bool = True,
            batch_size: int = 32,
            show_progress: bool = True,
            cache_dir: Optional[str] = None,
            use_prefixes: bool = False,
            query_prefix: str = "query: ",
            passage_prefix: str = "passage: ",
            max_seq_length: Optional[int] = None,  # np. 512 dla E5/GTE
    ):
        """
        Args:
            model_id: Model ID from SentenceTransformers or path to local model.
            device: Device to run the model on (e.g., 'cpu', 'cuda').
            normalize_embeddings: Whether to L2 normalize the embeddings.
            batch_size: Batch size for embedding.
            show_progress: Whether to show a progress bar during embedding.
            cache_dir: Directory to cache the model.
            use_prefixes: Whether to prepend prefixes to texts based on type.
            query_prefix: Prefix to prepend to query texts if use_prefixes is True.
            passage_prefix: Prefix to prepend to passage texts if use_prefixes is True.
            max_seq_length: Maximum sequence length for the model. If None, uses model default.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.cache_dir = cache_dir
        self.use_prefixes = use_prefixes
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.max_seq_length = max_seq_length

        logger.info(f"Loading embedder model '{model_id}' on device '{self.device}'")
        self.model = SentenceTransformer(model_id, device=self.device, cache_folder=cache_dir)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
            logger.info(f"Set max_seq_length to {max_seq_length}")
        logger.info("Model loaded successfully.")
