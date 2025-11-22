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
        if not contexts:
            logger.debug("No contexts found for claim")
            return None

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
            best_match_idx = matches[0][0]["id"]
            best_match_ctx = contexts[best_match_idx]

            # for post hop use citation_id, else idx + 1
            citation_id = best_match_ctx.get("citation_id")
            if citation_id is None:
                max_citation_id = max(
                    (ctx.get("citation_id", 0) for ctx in contexts),
                    default=0
                )
                citation_id = max_citation_id + 1
                contexts[best_match_idx].update({"citation_id": citation_id})
                logger.info(
                    f"Assigned NEW citation_id={citation_id} to contexts[{best_match_idx}]"
                )
            logger.info(
                f"Injected citation [{citation_id}] for claim: {claim_clean[:50]}..."
            )
            return [citation_id]

        logger.debug(f"No good citation match for claim: {claim_clean[:50]}...")
        return None

    def enrich_with_citations(
            self,
            answer: str,
            contexts: List[Dict[str, Any]],
    ) -> tuple[str, bool]:
        """
        Enrich answer by adding citations sentence-by-sentence.
        PRESERVES paragraph breaks (\n\n) in the original answer.
        PREVENTS overwriting existing citations and removes duplicates.

        Args:
            answer: The input answer to be enriched with citations.
            contexts: List of context dictionaries containing relevant information.

        Returns:
            (enriched_answer, enrichment_applied)
        """
        logger.debug(f"Input answer for citation enrichment (first 200 chars): {answer[:200]}...")

        # Split by sentences but PRESERVE separators (space vs \n\n)
        # Capture group (\s+) returns separators in the split result
        parts = re.split(r'(?<=[.!?])(\s+)', answer)

        if not parts:
            return answer, False

        enriched_parts = []
        any_injected = False

        for i, part in enumerate(parts):
            # Even indices are sentences, odd indices are separators
            if i % 2 == 1:
                # This is a separator (space or \n\n) - keep as-is
                enriched_parts.append(part)
                continue

            # This is a sentence - check for citations
            sentence = part.strip()
            if not sentence:
                enriched_parts.append(part)
                continue

            # CRITICAL: Check for citations in ORIGINAL part, not stripped
            # This prevents overwriting existing citations like [4][8][10]
            has_citation = bool(re.search(r'\[\d+\]', part))

            if has_citation:
                # Already has citations - preserve original completely
                logger.debug(f"Skipping sentence with existing citations: {sentence[:50]}...")
                enriched_parts.append(part)
                continue

            dummy_claim = Claim(
                text=sentence,
                claim_id=0,
                has_citations=False,
                citations=[],
            )

            injected = self.inject(dummy_claim, contexts)

            if injected:
                # Extract existing citations from this sentence (shouldn't happen due to has_citation check above, but be safe)
                existing_citations = re.findall(r'\[(\d+)\]', sentence)
                existing_ids = [int(cid) for cid in existing_citations]

                # Only add NEW citations (avoid duplicates like [3][3])
                new_citations = [c for c in injected if c not in existing_ids]

                if new_citations:
                    # Remove duplicates within new citations and sort
                    unique_new = sorted(set(new_citations))
                    # Add citation at the end of sentence
                    enriched_sentence = f"{sentence} [{','.join(map(str, unique_new))}]"
                    enriched_parts.append(enriched_sentence)
                    any_injected = True
                    logger.debug(f"Enriched: {sentence[:50]}... → added {unique_new} (filtered from {injected})")
                else:
                    # All citations already exist - don't add duplicates
                    logger.debug(f"Skipped: {sentence[:50]}... → all citations {injected} already present")
                    enriched_parts.append(part)
            else:
                # No citation match found - leave as-is
                enriched_parts.append(part)

        if any_injected:
            # Join all parts (sentences + original separators) - preserves \n\n!
            enriched_answer = "".join(enriched_parts)
            sentence_count = len([p for i, p in enumerate(parts) if i % 2 == 0 and p.strip()])
            logger.info(f"Citation enrichment applied to {sentence_count} sentences")
            logger.debug(f"Output answer (first 200 chars): {enriched_answer[:200]}...")
            return enriched_answer, True

        logger.debug("No citations were injected - returning original answer")
        return answer, False
