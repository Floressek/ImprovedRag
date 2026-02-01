from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BasePipeline(ABC):
    """Abstract base class for RAG pipelines."""

    @abstractmethod
    def answer(
            self,
            query: str,
            chat_history: Optional[List[Dict[str, str]]] = None,
            **kwargs: Any
    )-> Dict[str, Any]:
        """Generate answer for a query."""
        pass

    def answer_stream(
            self,
            query: str,
            chat_history: Optional[List[Dict[str, str]]] = None,
            **kwargs: Any
    ) -> Any:
        """Stream answer tokens for a query."""
        pass