import logging
from typing import Optional

import numpy as np
from src.ragx.retrieval.embedder.embedder import Embedder


logger = logging.getLogger(__name__)


class EmbeddingBatcher:
    """Helper class for batched embedding processing."""

    def __init__(
            self,
            embedder: Embedder,
            batch_size: int = 100,
            max_texts: Optional[int] = None,
    ):
        """
        Args:
            embedder: Embedder instance
            batch_size: Buffer size before encoding
            max_texts: Hard cap of processed texts (None = unlimited)
        """
        self.embedder = embedder
        self.batch_size = int(batch_size)
        self.max_texts = max_texts
        self._buffer: list[str] = []
        self._embeddings: list[list[float]] = []
        self._processed = 0  # Counter of processed texts

    def _remaining_quota(self) -> Optional[int]:
        if self.max_texts is None:
            return None
        return max(self.max_texts - (self._processed + len(self._buffer)), 0)

    def add(self, text: str) -> None:
        """Add single text; respects max_texts cap."""
        if self.max_texts is not None and self._processed >= self.max_texts:
            return
        if self.max_texts is not None and self._processed + len(self._buffer) >= self.max_texts:
            return

        self._buffer.append(text)
        if len(self._buffer) >= self.batch_size:
            self._process_buffer()

    def add_batch(self, texts: list[str]) -> None:
        """Add multiple texts; slices to remaining quota if needed."""
        if not texts:
            return
        if self.max_texts is None:
            for t in texts:
                self.add(t)
            return

        remaining = self._remaining_quota()
        if remaining is not None and remaining <= 0:
            return

        to_take = texts if remaining is None else texts[:remaining]
        for t in to_take:
            self.add(t)

    def _process_buffer(self) -> None:
        """Process current buffer."""
        if not self._buffer:
            return

        embs = self.embedder.embed_texts(
            self._buffer,
            convert_to_numpy=True,
            show_progress=False,
        )
        if isinstance(embs, np.ndarray):
            embs = embs.astype(np.float32, copy=False).tolist()

        self._embeddings.extend(embs)
        self._processed += len(self._buffer)
        self._buffer.clear()

        if self._processed % 1000 == 0:
            logger.info("Processed %d texts", self._processed)

    def finish(self) -> list[list[float]]:
        """Flush remaining buffer and return all embeddings (list of float lists)."""
        self._process_buffer()
        return self._embeddings

    def reset(self) -> None:
        """Reset state."""
        self._buffer.clear()
        self._embeddings.clear()
        self._processed = 0
