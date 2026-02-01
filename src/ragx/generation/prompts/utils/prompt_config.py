from dataclasses import dataclass


@dataclass
class PromptConfig:
    """Configuration for prompt building."""
    use_cot: bool = True
    include_metadata: bool = True
    strict_citations: bool = True
    detect_language: bool = True
    check_contradictions: bool = True
    confidence_scoring: bool = True
    think_tag_style: str = "qwen"  # "qwen", "none"