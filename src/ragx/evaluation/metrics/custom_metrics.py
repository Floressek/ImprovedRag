from typing import List, Dict, Any


def count_sources(sources: List[str]) -> int:
    """
    Count unique sources (URLs, doc IDs, etc.).

    Args:
        sources: List of source identifiers (URLs, doc IDs, etc.)

    Returns:
        Number of unique sources
    """
    return len(set(sources)) if sources else 0


def calculate_multihop_coverage(
        sub_queries: List[str],
        results_by_subquery: Dict[str, List[Any]],
) -> float:
    """
    Calculate multihop coverage: ratio of sub-queries with ≥1 retrieved doc.

    For multihop queries, this shows how well the retrieval covers all sub-queries.
    A score of 1.0 means every sub-query got at least one document.
    A score of 0.5 means only half the sub-queries got documents.

    Args:
        sub_queries: List of sub-queries from decomposition
        results_by_subquery: Dict mapping sub-query → retrieved results

    Returns:
        Coverage ratio (0.0 to 1.0), or 0.0 for non-multihop queries
    """
    if not sub_queries or len(sub_queries) <= 1:
        # Single query or no decomposition → not applicable
        return 0.0  # Coverage is N/A for non-multihop

    covered = 0
    for sq in sub_queries:
        results = results_by_subquery.get(sq)
        # Handle both formats: int (count) or list
        if isinstance(results, int):
            if results > 0:
                covered += 1

        elif results and len(results) > 0:
            covered += 1
    return covered / len(sub_queries)
