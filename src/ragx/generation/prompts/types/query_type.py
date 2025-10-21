from enum import Enum


class QueryType(Enum):
    FACTUAL = "factual"          # Simple fact lookup
    ANALYTICAL = "analytical"    # Requires analysis/comparison
    TEMPORAL = "temporal"        # Time-related questions
    QUANTITATIVE = "quantitative" # Numbers/statistics
    DEFINITIONAL = "definitional" # What is X?
    PROCEDURAL = "procedural"    # How to do X?
    COMPARATIVE = "comparative"  # Compare X and Y