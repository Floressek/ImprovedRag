from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Iterator, Optional, List, Dict, Any

from transformers import AutoTokenizer
from langchain_text_splitters import TokenTextSplitter

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)


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


class TextChunker:
    """
    strategy: "semantic" (LlamaIndex SemanticSplitter) | "token" (LangChain TokenTextSplitter)
    """

    def __init__(
            self,
            strategy: str = "semantic",
            chunk_size: int = 512,
            chunk_overlap: int = 96,
            min_chunk_size: int = 100,
            model_name_tokenizer: str = "thenlper/gte-multilingual-base",
            model_name_embedder: str = "thenlper/gte-multilingual-base",
            respect_sections: bool = True,
            breakpoint_percentile_thresh: int = 95,
            add_passage_prefix: bool = False,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        if strategy not in {"semantic", "token"}:
            raise ValueError("strategy must be 'semantic' or 'token'")

        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sections = respect_sections
        self.breakpoint_percentile_thresh = breakpoint_percentile_thresh
        self.add_passage_prefix = add_passage_prefix

        self._tok = AutoTokenizer.from_pretrained(model_name_tokenizer)
        self._token_splitter = TokenTextSplitter.from_huggingface_tokenizer(
            self._tok, chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=False
        )

        self._embed = HuggingFaceEmbedding(model_name=model_name_embedder)
        self._semantic = SemanticSplitterNodeParser(
            embed_model=self._embed,
            buffer_size=1,
            breakpoint_percentile_thresh=breakpoint_percentile_thresh,
        )

        self._section_re = re.compile(r'(?:^|\n)(={2,6})\s*(.+?)\s*\1\s*(?=\n|$)')

    def chunk_document(
            self,
            text: str,
            doc_id: str,
            doc_title: str,
            metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        sections = self._split_sections(text) if self.respect_sections else [text]

        chunk_texts: List[str] = []
        for sec_text in sections:
            if not sec_text.strip():
                continue
            if self.strategy == "semantic":
                for seg in self._semantic_split(sec_text):
                    if self._count_tokens(seg) > self.chunk_size:
                        for tdoc in self._token_splitter.create_documents([seg]):
                            chunk_texts.append(tdoc.page_content)
                    else:
                        chunk_texts.append(seg)
            else:
                for tdoc in self._token_splitter.create_documents([sec_text]):
                    chunk_texts.append(tdoc.page_content)

        if chunk_texts:
            if self._count_tokens(chunk_texts[-1]) < self.min_chunk_size and len(chunk_texts) >= 2:
                chunk_texts[-2] = f"{chunk_texts[-2]} {chunk_texts[-1]}".strip()
                chunk_texts.pop()

        if self.add_passage_prefix:
            chunk_texts = [f"passage: {c}" for c in chunk_texts]

        total = len(chunk_texts)
        out: List[Chunk] = []
        for pos, ctext in enumerate(chunk_texts):
            cid = self._generate_chunk_id(doc_id, pos)
            out.append(Chunk(
                id=cid,
                text=ctext,
                doc_id=doc_id,
                doc_title=doc_title,
                position=pos,
                total_chunks=total,
                token_count=self._count_tokens(ctext),
                char_count=len(ctext),
                metadata=metadata.copy(),
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
        nodes = self._semantic.get_nodes_from_documents([Document(text=text)])
        out: List[str] = []
        for n in nodes:
            try:
                seg = n.get_content(metadata_mode="none")
            except Exception:
                seg = getattr(n, "text", "")
            if seg and seg.strip():
                out.append(seg.strip())
        return out

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
        if not s:
            return 0
        return len(self._tok(s, add_special_tokens=False, truncation=False)["input_ids"])

    def _generate_chunk_id(self, doc_id: str, position: int) -> str:
        digest = hashlib.md5(f"{doc_id}_{position}".encode()).hexdigest()[:8]
        return f"chunk_{doc_id}_{position}_{digest}"
