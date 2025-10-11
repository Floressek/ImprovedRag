from dataclasses import dataclass
from typing import Optional, Self


@dataclass
class FileMetadata:
    """Metadata for a file."""
    file_path: str
    started_at: str
    articles_count: int
    chunks_count: int
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "started_at": self.started_at,
            "articles_count": self.articles_count,
            "chunks_count": self.chunks_count,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(**data)
