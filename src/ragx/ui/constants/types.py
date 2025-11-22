from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class PipelineConfig:
    """Pipeline configuration preset."""
    name: str
    description: str
    query_analysis_enabled: bool = True
    cot_enabled: bool = True
    reranker_enabled: bool = True
    cove_mode: str = "off"
    prompt_template: str = "auto"
    top_k: int = 10


@dataclass
class PipelineStep:
    """Represents a single pipeline step."""
    key: str
    message: str
    enabled: bool
    estimated_duration: float = 0.0  # in seconds


@dataclass
class StepTiming:
    """Timing estimates for pipeline steps based on configuration."""
    query_analysis: float = 0.0
    retrieval: float = 0.0
    reranking: float = 0.0
    generation: float = 0.0
    cove: float = 0.0

    @property
    def total(self) -> float:
        """Total estimated duration."""
        return sum([
            self.query_analysis,
            self.retrieval,
            self.reranking,
            self.generation,
            self.cove
        ])
