from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Optional

from src.ragx.retrieval.cove.constants.types import Claim
from src.ragx.retrieval.rerankers.reranker import Reranker

logger = logging.getLogger(__name__)


class CitationInjector:
    """Inject missing citations for verified claims (minimal damage control)."""

    def __init__(self, reranker: Reranker):
        self.reranker = reranker

    def inject(
            self,
            claim: Claim,
            contexts: List[Dict[str, Any]]
    ) -> Optional[List[int]]:
        """
        Find best matching context for a claim and return citations ID
        If claim is verified, but has no citations, inject it.
        """
        # Optional fix for correcting citation injection FIXME delete this comment or code if buggy
        # Remove existing citations from claim text for matching
        claim_clean = re.sub(r'\[\d+\]', '', claim.text).strip()

        documents = [
            {
                "id": i,
                "text": ctx.get("text", ""),
                "doc_title": ctx.get("doc_title", "Unknown"),
            }
            for i, ctx in enumerate(contexts)
        ]

        matches = self.reranker.rerank(
            query=claim_clean,
            documents=documents,
            top_k=3,
            text_field="text",
        )

        if matches and matches[0][1] > 0.6:
            best_match_id = matches[0][0]["id"]
            logger.info(
                f"Injected citation [{best_match_id+1}] for claim: {claim_clean[:50]}..."
            )
            return [best_match_id + 1]

        logger.debug(f"No good citation match for claim: {claim_clean[:50]}...")
        return None


