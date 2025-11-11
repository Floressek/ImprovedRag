from enum import Enum


class CoVeStatus(str, Enum):
    ALL_VERIFIED = "all_verified"
    MISSING_EVIDENCE = "missing_evidence"
    MISSING_CITATIONS = "missing_citations"
    LOW_CONFIDENCE = "low_confidence"
    CRITICAL_FAILURE = "critical_failure"
    NO_CLAIMS = "no_claims"
    SKIPPED = "skipped"

    def is_correction_needed(self) -> bool:
        """Check if this status requires answer correction (not just citation injection)."""
        return self in [
            CoVeStatus.MISSING_EVIDENCE,
            CoVeStatus.LOW_CONFIDENCE,
            CoVeStatus.CRITICAL_FAILURE,
        ]

    def is_metadata_only(self) -> bool:
        """Check if this status is just metadata/info (no correction needed)."""
        return self in [
            CoVeStatus.ALL_VERIFIED,
            CoVeStatus.MISSING_CITATIONS,  # Just formatting issue
            CoVeStatus.NO_CLAIMS,
            CoVeStatus.SKIPPED,
        ]