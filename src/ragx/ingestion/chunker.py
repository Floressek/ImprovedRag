from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

from transformers import AutoTokenizer
from langchain_text_splitters import TokenTextSplitter

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    id: str
    text: str
    doc_id: str
    doc_title: str
    position: int  # Position in document (0-based)
    total_chunks: int  # Total chunks in document
    token_count: int
    char_count: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert Chunk to dictionary for storage."""
        return {
            "id": self.id,
            "text": self.text,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "position": self.position,
            "total_chunks": self.total_chunks,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "metadata": self.metadata,
        }

    @property
    def is_first(self) -> bool:
        """Check if this is the first chunk in the document."""
        return self.position == 0

    @property
    def is_last(self) -> bool:
        """Check if this is the last chunk in the document."""
        return self.position == self.total_chunks - 1

class TextChunker:
    """Text chunker with semantic boundary awareness."""

    def __init__(
            self,
            chunk_size: int = 768,
            chunk_overlap: int = 96,
            min_chunk_size: int = 100,
            tokenizer: str = "word",  # 'word' or 'char'
            respect_sentences: bool = True,
    ):
        """Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens/chars
            chunk_overlap: Overlap between chunks in tokens/chars
            min_chunk_size: Minimum chunk size to create
            tokenizer: Type of tokenization ('word' or 'char')
            respect_sentences: Try to preserve sentence boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.tokenizer = tokenizer
        self.respect_sentences = respect_sentences

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    def chunk_document(
            self,
            text: str,
            doc_id: str,
            doc_title: str,
            metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """Chunk a document into overlapping pieces.

        Args:
            doc_id: Unique document identifier
            doc_title: Document title
            text: Full document text
            metadata: Additional metadata

        Yields:
            CList of Chunk objects
        """
        if not text or not text.strip():
            return []

        metadate = metadata or {}
        chunks = []

        if self.respect_sentences:
            sentences = self._split_sentences(text)

