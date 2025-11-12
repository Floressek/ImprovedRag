from typing import Tuple, List, Any, Dict
import re
import logging

logger = logging.getLogger(__name__)

def remap_citations(
        answer: str,
        citation_mapping: Dict[int, str],
        contexts: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Remap citations to [1], [2], [3] and reorganize contexts:
    - Cited docs first (with citation_id)
    - Uncited docs after (for CoVe, without citation_id)
    """
    citations_used = set()
    for match in re.finditer(r'\[(\d+)\]', answer):
        citations_used.add(int(match.group(1)))

    if not citations_used:
        logger.warning("No citations found in answer")
        return answer, contexts

    logger.info(f"Found {len(citations_used)} unique citations: {sorted(citations_used)}")

    # Map: real_doc_id -> new_citation_id (1-based sequential)
    used_docs = {}
    new_id = 1

    for prompt_cit_id in sorted(citations_used):
        real_doc_id = citation_mapping.get(prompt_cit_id)
        if real_doc_id and real_doc_id not in used_docs:
            used_docs[real_doc_id] = new_id
            logger.debug(f"Mapped [{prompt_cit_id}] -> [{new_id}] (doc: {real_doc_id[:12]})")
            new_id += 1

    # Remap citations in answer
    def replace_citation(match):
        old_id = int(match.group(1))
        real_doc_id = citation_mapping.get(old_id)
        if real_doc_id and real_doc_id in used_docs:
            return f"[{used_docs[real_doc_id]}]"
        logger.warning(f"Citation [{old_id}] not found in mapping")
        return match.group(0)

    remapped_answer = re.sub(r'\[(\d+)\]', replace_citation, answer)

    # Cited first, then uncited (all preserved)
    cited_contexts = []
    uncited_contexts = []

    for ctx in contexts:
        ctx_copy = dict(ctx)
        doc_id = ctx.get("id")

        if doc_id in used_docs:
            ctx_copy["citation_id"] = used_docs[doc_id]
            cited_contexts.append(ctx_copy)
        else:
            # No citation_id for uncited (CoVe)
            uncited_contexts.append(ctx_copy)

    # Sort cited by citation_id for clean output
    cited_contexts.sort(key=lambda x: x.get("citation_id", 999))

    # Final order: cited first, then uncited
    reorganized_contexts = cited_contexts + uncited_contexts

    logger.info(
        f"Remapped citations: {len(cited_contexts)} cited, "
        f"{len(uncited_contexts)} uncited (for CoVe), "
        f"{len(reorganized_contexts)} total"
    )

    return remapped_answer, reorganized_contexts