from enum import Enum


class CoVeStatus(str, Enum):
    ALL_VERIFIED = "all_verified"
    MISSING_EVIDENCE = "missing_evidence"
    LOW_CONFIDENCE = "low_confidence"
    CRITICAL_FAILURE = "critical_failure"
    NO_CLAIMS = "no_claims"
    SKIPPED = "skipped"