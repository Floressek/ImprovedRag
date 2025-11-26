import logging
import math
import statistics
from typing import List, Tuple

logger = logging.getLogger(__name__)


def safe_std(values: List[float]) -> float:
    """
    Calculate standard deviation, handling edge cases.

    Args:
        values: List of numeric values

    Returns:
        Standard deviation, or 0.0 if not enough data
    """
    if len(values) < 2:
        return 0.0
    try:
        return statistics.stdev(values)
    except statistics.StatisticsError:
        return 0.0


def calculate_ci(
        values: List[float],
        confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate confidence interval.

    Uses formula: CI = mean ± z * (std / sqrt(n))
    For 95% CI, z = 1.96

    Args:
        values: List of metric values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) < 2:
        # Not enough data for CI - need at least 2 samples
        # Return (0, 0) to indicate invalid CI rather than misleading equal bounds
        logger.warning(f"Cannot calculate CI with n={len(values)} samples (need n>=2)")
        return (0.0, 0.0)

    try:
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        n = len(values)

        # Z-score for 95% CI
        z_score = 1.96

        # Margin of error
        margin = z_score * (std_val / math.sqrt(n))

        return (mean_val - margin, mean_val + margin)

    except (statistics.StatisticsError, ZeroDivisionError):
        mean_val = statistics.mean(values) if values else 0.0
        return (mean_val, mean_val)
