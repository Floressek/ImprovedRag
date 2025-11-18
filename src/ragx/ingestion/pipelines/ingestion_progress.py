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

    processed_files: Set[str] = field(default_factory=set)
    file_metadata: dict[str, dict] = field(default_factory=dict)

    current_file: Optional[str] = None
    current_folder: Optional[str] = None

    chunk_size: Optional[int] = None
    chunk_strategy: Optional[str] = None
    embedding_model: Optional[str] = None
    collection_name: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        data = asdict(self)
        data['processed_files'] = sorted(list(self.processed_files))
        return data

    @classmethod
    def from_dict(cls, data: dict) -> IngestionProgress:
        """Deserialize from dict."""
        data['processed_files'] = set(data.get('processed_files', []))
        data['file_metadata'] = data.get('file_metadata', {})
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save progress to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.debug(f"Progress saved to {path}")

    @classmethod
    def load(cls, path: Path) -> Optional[IngestionProgress]:
        """Load progress from JSON file."""
        if not path.exists():
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded progress from {path}")
            return cls.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load progress from {path}: {e}")
            return None

    def start_file(self, file_path: str) -> None:
        """Mark the start of processing a new file."""
        self.current_file = file_path
        self.current_folder = str(Path(file_path).parent)

        if file_path not in self.file_metadata:
            self.file_metadata[file_path] = {
                "file_path": file_path,
                "started_at": datetime.now().isoformat(),
                "articles_count": 0,
                "chunks_count": 0,
                "completed_at": None,
            }
        self.last_updated = datetime.now().isoformat()

    def complete_file(self, file_path: str) -> None:
        """Mark file as completed."""
        self.processed_files.add(file_path)
        if file_path in self.file_metadata:
            self.file_metadata[file_path]["completed_at"] = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()
        logger.info(f"✓ File completed: {Path(file_path).name}")

    def add_article(self, file_path: str, num_chunks: int) -> None:
        """Increment article/chunk counters."""
        self.total_articles += 1
        self.total_chunks += num_chunks

        if file_path in self.file_metadata:
            self.file_metadata[file_path]["articles_count"] += 1
            self.file_metadata[file_path]["chunks_count"] += num_chunks

        self.last_updated = datetime.now().isoformat()

    def is_file_processed(self, file_path: str) -> bool:
        return file_path in self.processed_files

    def get_summary(self) -> dict:
        completed = len(self.processed_files)
        in_progress = 1 if self.current_file and self.current_file not in self.processed_files else 0

        return {
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "total_articles": self.total_articles,
            "total_chunks": self.total_chunks,
            "total_batches": self.total_batches,
            "files_completed": completed,
            "files_in_progress": in_progress,
            "current_file": Path(self.current_file).name if self.current_file else None,
            "current_folder": self.current_folder,
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_strategy": self.chunk_strategy,
                "embedding_model": self.embedding_model,
                "collection": self.collection_name,
            }
        }