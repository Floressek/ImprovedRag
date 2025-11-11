from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Optional

from src.ragx.retrieval.cove.constants.types import Claim
from src.ragx.retrieval.cove.tools.sentence_splitter import split_sentences
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

        Args:
            claim: The claim object to be processed.
            contexts: List of context dictionaries containing relevant information.

        Returns:
            List of citation IDs injected into the claim.
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

    def enrich_with_citations(
            self,
            answer: str,
            contexts: List[Dict[str, Any]],
    ) -> tuple[str, bool]:
        """
        Enrich answer by adding citations sentence-by-sentence.

        Args:
            answer: The input answer to be enriched with citations.
            contexts: List of context dictionaries containing relevant information.

        Returns:
            (enriched_answer, enrichment_applied)
        """
        sentences = split_sentences(answer)

        if not sentences:
            return answer, False

        enriched_sentences = []
        any_injected = False

        for sentence in sentences:
            has_citation = bool(re.search(r'\[\d+\]', sentence))

            if has_citation:
                enriched_sentences.append(sentence)
                continue

            dummy_claim = Claim(
                text=sentence,
                claim_id=0,
                has_citations=False,
                citations=[],
            )

            injected = self.inject(dummy_claim, contexts)

            if injected:
                # add at the end
                enriched_sentence = f"{sentence} [{','.join(map(str, injected))}]"
                enriched_sentences.append(enriched_sentence)
                any_injected = True
                logger.debug(f"Enriched: {sentence[:50]}... → added {injected}")
            else:
                # didnt work leave as it is
                enriched_sentences.append(sentence)

        if any_injected:
            enriched_answer = " ".join(enriched_sentences)
            logger.info(f"Citation enrichment applied to {len(sentences)} sentences")
            return enriched_answer, True

        return answer, False


