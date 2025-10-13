from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class Enhancer(ABC):
    """Abstract base class for pipeline enhancers."""

    @abstractmethod
    def process(
            self,
            query: str,
            results: List[tuple]
    ) -> List[tuple]:
        """Process retrival results."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the enhancer."""
        pass
