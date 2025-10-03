from __future__ import annotations

import uuid
import hashlib
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator, Optional, List, Dict, Any, Callable

from transformers import AutoTokenizer
from langchain_text_splitters import TokenTextSplitter

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from llama_index.core.schema import MetadataMode

from .constants.chunker_types import ChunkingStrategy
from .utils.strip_header import extract_section_header, strip_leading_header

logger = logging.getLogger(__name__)


# FIXME half of the utils functions go to the separate file
@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    doc_title: str
    position: int
    total_chunks: int
    token_count: int
    char_count: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
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
        return self.position == 0

    @property
    def is_last(self) -> bool:
        return self.position == self.total_chunks - 1


# tgo to utils
def _stable_chunk_id(doc_id: str, position: str, text: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(doc_id.encode("utf-8", "ignore"))
    h.update(b"::")
    h.update(str(position).encode())
    h.update(b"::")
    h.update(text.encode("utf-8", "ignore"))
    return str(uuid.UUID(bytes=h.digest()))


class TextChunker:
    """
    Text chunking with semantic or token-based strategies.

    Strategies:
        - semantic: LlamaIndex SemanticSplitter (context-aware)
        - token: LangChain TokenTextSplitter (fixed-size)
    """

    def __init__(
            self,
            strategy: str = ChunkingStrategy.SEMANTIC,
            chunk_size: int = 450,
            chunk_overlap: int = 128,
            min_chunk_size: int = 150,
            max_chunk_size: int = 512,
            model_name_tokenizer: str = "Alibaba-NLP/gte-multilingual-base",
            model_name_embedder: str = "Alibaba-NLP/gte-multilingual-base",
            respect_sections: bool = True,
            breakpoint_percentile_thresh: int = 80,
            buffer_size: int = 3,
            add_passage_prefix: bool = False,
            trust_remote_code: bool = True,
            context_tail_tokens: int = 0,
    ):
        """
        Initialize text chunker.

        Args:
            strategy: Chunking strategy (semantic or token)
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size (smaller chunks get merged)
            max_chunk_size: Maximum chunk size (larger chunks get split)
            model_name_tokenizer: Tokenizer model
            model_name_embedder: Embedding model for semantic chunking
            respect_sections: Whether to respect section boundaries
            breakpoint_percentile_thresh: Semantic breakpoint threshold (75-90)
            buffer_size: Buffer size for semantic chunking (2-5)
            add_passage_prefix: Add "passage: " prefix to chunks
            trust_remote_code: Trust remote code in models
            context_tail_tokens: Tokens to keep as context tail (not chunked)
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        if min_chunk_size >= chunk_size:
            raise ValueError(f"min_chunk_size ({min_chunk_size}) must be < chunk_size ({chunk_size})")
        if strategy not in {"semantic", "token"}:
            raise ValueError("strategy must be 'semantic' or 'token'")

        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.respect_sections = respect_sections
        self.breakpoint_percentile_thresh = breakpoint_percentile_thresh
        self.buffer_size = buffer_size
        self.add_passage_prefix = add_passage_prefix
        self.context_tail_tokens = max(0, int(context_tail_tokens))

        logger.info(f"Loading tokenizer: {model_name_tokenizer}")
        self._tok = AutoTokenizer.from_pretrained(
            model_name_tokenizer,
            trust_remote_code=trust_remote_code
        )

        model_ctx = getattr(self._tok, "model_max_length", 512) or 512
        safe_cap = int(model_ctx * 0.9)
        if self.max_chunk_size > safe_cap:
            self.max_chunk_size = safe_cap

        self._token_splitter = TokenTextSplitter.from_huggingface_tokenizer(
            self._tok,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=False
        )

        self._embed = None
        self._semantic = None
        if strategy == ChunkingStrategy.SEMANTIC:
            logger.info(f"Loading embedding model for semantic chunking: {model_name_embedder}")
            self._embed = HuggingFaceEmbedding(
                model_name=model_name_embedder,
                trust_remote_code=trust_remote_code
            )
            self._semantic = SemanticSplitterNodeParser(
                embed_model=self._embed,
                buffer_size=buffer_size,
                breakpoint_percentile_threshold=breakpoint_percentile_thresh,
            )

        self._section_re = re.compile(r"(?:^|\n)(={2,6})\s*(.+?)\s*\1\s*(?=\n|$)")
        self._sentence_re = re.compile(r"(?<=[.!?])\s+(?=[A-ZŁĄĆĘŃÓŚŹŻ])")  # Polish sentence boundaries

        self._count_tokens_cached: Callable[[str], int] = self._make_token_counter(self._tok)

        logger.info(
            f"TextChunker initialized: strategy={strategy}, chunk_size={chunk_size}, overlap={chunk_overlap}, "
            f"min={min_chunk_size}, max={self.max_chunk_size}, buffer={self.buffer_size}, "
            f"breakp={self.breakpoint_percentile_thresh}"
        )

    def _make_token_counter(self, tok) -> Callable[[str], int]:
        """
        Create a cached token counting function.
        Uses LRU cache to avoid redundant tokenization.
        """

        @lru_cache(maxsize=65536)
        def _counter(s: str) -> int:
            if not s:
                return 0
            return len(tok(s, add_special_tokens=False, truncation=False)["input_ids"])

        return _counter

    def clear_caches(self) -> None:
        """
        Clear internal caches (e.g. token counting).
        """
        self._count_tokens_cached.cache_clear()

    def chunk_document(
            self,
            text: str,
            doc_id: str,
            doc_title: str,
            metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        """
        Chunk a single document into smaller pieces.

        Args:
            text: Document text
            doc_id: Document ID
            doc_title: Document title
            metadata: Additional metadata

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        self.clear_caches()

        metadata = metadata or {}
        sections = self._split_sections(text) if self.respect_sections else [text]

        chunk_texts: List[str] = []
        for sec_text in sections:
            if not sec_text.strip():
                continue
            if self.strategy == ChunkingStrategy.SEMANTIC:
                for seg in self._semantic_split(sec_text):
                    token_count = self._count_tokens(seg)

                    # If segment too large, split with token splitter
                    if token_count > self.max_chunk_size:
                        for tdoc in self._token_splitter.create_documents([seg]):
                            chunk_texts.append(tdoc.page_content)
                    else:
                        chunk_texts.append(seg)
            else:
                # Currently aside from semantic i only consider token-based
                for tdoc in self._token_splitter.create_documents([sec_text]):
                    chunk_texts.append(tdoc.page_content)

        # Drop micro fragments
        chunk_texts = [c for c in chunk_texts if self._count_tokens(c) > 5]

        chunk_texts = self._merge_small_chunks(chunk_texts)

        # optional tail-context window
        if self.context_tail_tokens > 0:
            chunk_texts = self._add_context_window(chunk_texts, self.context_tail_tokens)

        if self.add_passage_prefix:
            chunk_texts = [f"passage: {c}" for c in chunk_texts]

        total = len(chunk_texts)
        out: List[Chunk] = []

        for pos, ctext in enumerate(chunk_texts):
            cid = _stable_chunk_id(doc_id, pos, ctext)
            extended_metadata = self._enrich_metadata(
                base_metadata=metadata,
                chunk_text=ctext,
                full_text=text,
                position=pos,
                total=total
            )
            out.append(Chunk(
                id=cid,
                text=ctext,
                doc_id=doc_id,
                doc_title=doc_title,
                position=pos,
                total_chunks=total,
                token_count=self._count_tokens(ctext),
                char_count=len(ctext),
                metadata=extended_metadata,
            ))
        return out

    def chunk_documents(
            self,
            documents: Iterator[Dict[str, Any]],
            text_field: str = "text",
            id_field: str = "id",
            title_field: str = "title",
    ) -> Iterator[Chunk]:
        for doc in documents:
            text = str(doc.get(text_field, "") or "")
            doc_id = str(doc.get(id_field, "") or "")
            doc_title = str(doc.get(title_field, "") or "")
            meta = {k: v for k, v in doc.items() if k not in [text_field, id_field, title_field]}
            for ch in self.chunk_document(text, doc_id, doc_title, meta):
                yield ch

    def _semantic_split(self, text: str) -> List[str]:
        if not self._semantic:
            raise RuntimeError("Semantic splitter not initialized")
        nodes = self._semantic.get_nodes_from_documents([Document(text=text)])
        out: List[str] = []
        for n in nodes:
            try:
                seg = n.get_content(metadata_mode=MetadataMode.NONE)
            except (AttributeError, KeyError, TypeError):
                seg = getattr(n, "text", "")
            if seg and seg.strip():
                section = extract_section_header(seg)
                if section:
                    seg = strip_leading_header(seg)
                out.append(seg.strip())
        return out

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return []
        merged: List[str] = []
        buffer = chunks[0]
        for chunk in chunks[1:]:
            bt = self._count_tokens(buffer)
            ct = self._count_tokens(chunk)
            combined = bt + ct
            should_merge = False

            # Should-merge checks
            if bt < self.min_chunk_size:
                should_merge = True
            elif (bt < int(self.chunk_size * 0.7)) and (ct < int(self.chunk_size * 0.7)) and (
                    combined < self.max_chunk_size):
                should_merge = True

            if should_merge and combined < self.max_chunk_size:
                buffer = f"{buffer} {chunk}".strip()
            else:
                merged.append(buffer)
                buffer = chunk
        if buffer:
            merged.append(buffer)
        # if the last chunk is too small, merge it with the previous one
        if len(merged) >= 2 and self._count_tokens(merged[-1]) < self.min_chunk_size:
            merged[-2] = f"{merged[-2]} {merged[-1]}".strip()
            merged.pop()
        logger.debug(f"Merged {len(chunks)} chunks into {len(merged)}")
        return merged

    def _add_context_window(self, chunks: List[str], tail_tokens: int) -> List[str]:
        if not chunks or tail_tokens <= 0:
            return chunks
        out: List[str] = []
        prev_tail_ids: List[int] = []
        for c in chunks:
            ids = self._tok(c, add_special_tokens=False, truncation=False)["input_ids"]
            if prev_tail_ids:
                combined = prev_tail_ids + ids
                text_with_tail = self._tok.decode(combined, skip_special_tokens=True).strip()
                out.append(text_with_tail)
            else:
                out.append(c)
            prev_tail_ids = ids[-tail_tokens:] if len(ids) > tail_tokens else ids
        return out

    def _calculate_completeness(self, text: str) -> float:
        score = 0.0
        if text and text[0].isupper():
            score += 0.25
        if text and text.rstrip().endswith(('.', '!', '?')):
            score += 0.25
        sentence_count = len(self._sentence_re.split(text))
        if sentence_count >= 2:
            score += 0.25
        tok = self._count_tokens(text)
        if self.min_chunk_size <= tok <= self.max_chunk_size:
            score += 0.25
        return score

    def _enrich_metadata(
            self,
            base_metadata: dict,
            chunk_text: str,
            full_text: str,
            position: int,
            total: int,
    ) -> dict:
        enhanced = {
            **base_metadata,
            "chunk_strategy": self.strategy,
            "chunk_size_config": self.chunk_size,
            "doc_length": len(full_text),
        }
        section = extract_section_header(chunk_text)
        if section:
            enhanced["section"] = section
        enhanced["has_numbers"] = bool(re.search(r"\d+", chunk_text))
        enhanced["has_dates"] = bool(re.search(r"\b\d{3,4}\b", chunk_text))
        enhanced["has_list"] = bool(re.search(r"(?:^|\n)\s*[\d\-\*•]\s+", chunk_text))
        enhanced["position_ratio"] = str(position / max(total - 1, 1))
        enhanced["completeness_score"] = str(self._calculate_completeness(chunk_text))
        return enhanced

    def _split_sections(self, text: str) -> List[str]:
        idxs = [m.start() for m in self._section_re.finditer(text)]
        if not idxs:
            return [text]
        parts: List[str] = []
        starts = [0] + idxs
        ends = idxs[1:] + [len(text)]
        for s, e in zip(starts, ends):
            part = text[s:e].strip()
            if part:
                parts.append(part)
        return parts

    def _count_tokens(self, s: str) -> int:
        return self._count_tokens_cached(s)
