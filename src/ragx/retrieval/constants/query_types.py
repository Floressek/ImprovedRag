from enum import Enum
from typing import Dict


class QueryType(str, Enum):
    """
    Types of queries based on their retrieval and reasoning patterns.
    """
    VERIFICATION = "verification"
    COMPARISON = "comparison"
    SIMILARITY = "similarity"
    CHAINING = "chaining"
    TEMPORAL = "temporal"
    AGGREGATION = "aggregation"
    SUPERLATIVE = "superlative"
    SIMPLE = "simple"


# Query-type-specific weights RERANK
# Lower weight (0.2-0.3): trust local sub-query reranking more (independent analysis needed)
# Higher weight (0.6-0.7): trust global reranking more (holistic view needed)
QUERY_TYPE_WEIGHTS: Dict[str, float] = {
    QueryType.VERIFICATION: 0.2,      # Sub-queries gather facts independently
    QueryType.COMPARISON: 0.25,       # Each entity analyzed separately
    QueryType.SIMILARITY: 0.25,       # Characteristics of each entity are independent
    QueryType.CHAINING: 0.6,          # The final answer must fit the whole chain
    QueryType.TEMPORAL: 0.6,          # Temporal sequence needs global coherence
    QueryType.AGGREGATION: 0.5,       # Balance: collect locally, aggregate globally
    QueryType.SUPERLATIVE: 0.65,      # Global perspective needed to pick "the best"
    QueryType.SIMPLE: 0.7,            # Shouldn't be multihop, but if it is, prefer global
}

QUERY_TYPE_DESCRIPTIONS: Dict[str, str] = {
    QueryType.VERIFICATION: "Queries that check the truth of a statement or claim",
    QueryType.COMPARISON: "Queries comparing two or more entities on specific attributes",
    QueryType.SIMILARITY: "Queries finding commonalities or similarities between entities",
    QueryType.CHAINING: "Multi-hop queries linking entities through intermediate steps",
    QueryType.TEMPORAL: "Queries about events in time ranges or sequences",
    QueryType.AGGREGATION: "Queries requiring counting, summing, or aggregating information",
    QueryType.SUPERLATIVE: "Queries finding the best/worst/most/least with constraints",
    QueryType.SIMPLE: "Simple factual lookups or definitions",
}


def get_query_type_weight(query_type: str, default_weight: float) -> float:
    """
    Get the global rerank weight for a specific query type.

    Args:
        query_type: The query type string
        default_weight: Default weight if query type not found

    Returns:
        Weight value between 0.0 and 1.0
    """
    return QUERY_TYPE_WEIGHTS.get(query_type, default_weight)


def get_query_type_description(query_type: str) -> str:
    """
    Get description of a query type.

    Args:
        query_type: The query type string

    Returns:
        Description string
    """
    return QUERY_TYPE_DESCRIPTIONS.get(query_type, "Unknown query type")

