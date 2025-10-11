from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)
@dataclass
class IngestionProgress:
    """Tracks ingestion progress for resume capability."""

    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    total_articles: int = 0
    total_chunks: int = 0
    total_batches: int = 0

    processed_file: Set[str] = field(default_factory=set)
    processed_article_ids: Set[str] = field(default_factory=set)

    file_metadata: dict[str, dict] = field(default_factory=dict)

    current_file: Optional[str] = None
    current_folder: Optional[str] = None

    last_file: Optional[str] = None
    last_article_id: Optional[str] = None
    last_article_title: Optional[str] = None

    chunk_size: Optional[int] = None
    chunk_strategy: Optional[str] = None
    embedding_model: Optional[str] = None
    collection_name: Optional[str] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data['processed_file'] = sorted(list(self.processed_file))
        data['processed_article_ids'] = sorted(list(self.processed_article_ids))
        return data

    @classmethod
    def from_dict(cls, data: dict) -> IngestionProgress:
        data['processed_file'] = set(data.get('processed_file', []))
        data['processed_article_ids'] = set(data.get('processed_article_ids', []))
        data['file_metadata'] = data.get('file_metadata', {})
        return cls(**data)

    def start_file(self, file_path: str) -> None:
        """Mark the start of processing a new file."""
        folder = str(Path(file_path).parent)
        self.current_file = file_path
        self.current_folder = folder

        if file_path in self.file_metadata:
            self.file_metadata[file_path] = {
                "file_path": file_path,
                "started_at": datetime.now().isoformat(),
                "articles_count": 0,
                "chunks_count": 0,
                "completed_at": None,
            }
        self.last_updated = datetime.now().isoformat()

    def complete_file(self, file_path: str) -> None:
        """Mark the current file as completed."""
        if file_path in self.file_metadata:
            self.file_metadata[file_path]["completed_at"] = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()

    def mark_article_processed(self, article_id: str, article_title: str, num_chunks: int, file_path: Optional[str]) -> None:
        """Mark an article as processed."""
        self.processed_article_ids.add(article_id)
        self.last_article_id = article_id
        self.last_article_title = article_title
        self.total_articles += 1
        self.total_chunks += num_chunks

        if file_path and file_path in self.file_metadata:
            self.file_metadata[file_path]["articles_count"] += 1
            self.file_metadata[file_path]["chunks_count"] += num_chunks

        self.last_updated = datetime.now().isoformat()

