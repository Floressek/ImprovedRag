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
            model_id: str = "Alibaba-NLP/gte-multilingual-base",
            device: Optional[str] = None,
            normalize_embeddings: bool = True,
            batch_size: int = 32,
            show_progress: bool = True,
            cache_dir: Optional[str] = None,
            use_prefixes: bool = False,
            query_prefix: str = "query: ",
            passage_prefix: str = "passage: ",
            max_seq_length: Optional[int] = None,
            trust_remote_code: bool = True,
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
            trust_remote_code: Whether to trust remote code in model.
        """
        self.model_id = model_id
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.use_prefixes = use_prefixes
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Auto-selected device: %s", self.device)
        else:
            self.device = device
            logger.info("Using specified device: %s", self.device)

        logger.info("Loading embedder model: %s on %s", model_id, self.device)
        self.model = SentenceTransformer(
            model_id,
            device=self.device,
            cache_folder=cache_dir,
            trust_remote_code=trust_remote_code,  # Dodano tutaj
        )

        if max_seq_length is not None:
            try:
                self.model.max_seq_length = int(max_seq_length)
                logger.info("Set model.max_seq_length = %d", self.model.max_seq_length)
            except Exception as e:
                logger.warning("Could not set max_seq_length (%s): %s", max_seq_length, e)

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info("Embedder initialized with dimension: %d", self.embedding_dim)

    def embed_texts(
            self,
            texts: list[str],
            batch_size: Optional[int] = None,
            convert_to_numpy: bool = True,
            show_progress: Optional[bool] = None,
            add_prefix: bool = False,
            prefix: Optional[str] = None,
    ) -> Union[np.ndarray, list[list[float]]]:
        """
        Embed a list of texts.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for embedding. Defaults to self.batch_size.
            convert_to_numpy: Whether to return embeddings as a numpy array.
            show_progress: Whether to show a progress bar. Defaults to self.show_progress.
            add_prefix: Whether to add a prefix based on a text type.
            prefix: Specific prefix to add if add_prefix is True.

        Returns:
            Embeddings as a numpy array or list of lists.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32) if convert_to_numpy else []

        batch_size = batch_size or self.batch_size
        show_progress = show_progress if show_progress is not None else self.show_progress

        # Prefix handling
        if add_prefix:
            eff_prefix = prefix if prefix is not None else (self.passage_prefix if self.passage_prefix else "")
            if eff_prefix:
                texts = [eff_prefix + text for text in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
        )

        if convert_to_numpy:
            # ensure float32 dtype
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.asarray(embeddings, dtype=np.float32)
            else:
                embeddings = embeddings.astype(np.float32, copy=False)
            return embeddings

        # list of lists
        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(np.float32, copy=False).tolist()
        # list already - maybe TENSOR HERE?
        return embeddings

    def embed_query(
            self,
            query: str,
            normalize: Optional[bool] = None,
    ) -> list[float]:
        """Embed a single query text (applies query prefix if enabled)."""
        normalize = self.normalize_embeddings if normalize is None else normalize
        q = (self.query_prefix + query) if self.use_prefixes and self.query_prefix else query

        emb = self.model.encode(
            q,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if isinstance(emb, np.ndarray):
            return emb.astype(np.float32, copy=False).tolist()

        # Fallback
        return [float(x) for x in emb]

    def embed_documents(
            self,
            documents: list[dict],
            text_field: str = "text",
            batch_size: Optional[int] = None
    ) -> tuple[list[list[float]], list[dict]]:
        """Embed documents and return vectors with original docs (filtered for non-empty text)."""

        texts: list[str] = []
        valid_docs: list[dict] = []

        for doc in documents:
            text = str(doc.get(text_field, "")).strip()
            if text:
                texts.append(text)
                valid_docs.append(doc)

        if not texts:
            return [], []

        embs = self.embed_texts(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress=False,
            add_prefix=True,  # Documents treated as passages
            prefix=None,  # passage_prefix if self.use_prefixes else None
        )

        if isinstance(embs, np.ndarray):
            embs = embs.astype(np.float32, copy=False).tolist()

        return embs, valid_docs

    def get_dimension(self) -> int:
        return self.embedding_dim

    def warmup(self) -> None:
        """Warm up with a tiny call."""
        _ = self.embed_query("warmup")
        logger.info("Embedder warmup complete")


