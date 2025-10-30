from __future__ import annotations

import logging
import json

import yaml
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.ragx.generation.inference import LLMInference
from src.ragx.retrieval.analyzers.linguistic_analyzer import LinguisticAnalyzer, LinguisticFeatures
from src.ragx.retrieval.rewriters.tools.parse import safe_parse
from src.ragx.utils.settings import settings
from src.ragx.utils.model_registry import model_registry

logger = logging.getLogger(__name__)


class AdaptiveQueryRewriter:
    """LLM-based query rewriter with linguistic analysis context."""

    def __init__(
            self,
            llm: Optional[LLMInference] = None,
            analyzer: Optional[LinguisticAnalyzer] = None,
            prompts_path: Optional[Path] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            verify_before_retrieval: bool = True,
            enabled: Optional[bool] = True,
    ):
        """
        Initialize AdaptiveQueryRewriter.

        Args:
            llm: LLMInference instance
            analyzer: LinguisticAnalyzer instance
            prompts_path: Path to prompts YAML file
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            verify_before_retrieval: Verify query before retrieval
            enabled: Enable query rewriter
        """
        self.max_tokens = max_tokens or settings.rewrite.max_tokens
        self.temperature = temperature or settings.rewrite.temperature
        self.enabled = enabled or settings.rewrite.enabled
        self.verify_before_retrieval = verify_before_retrieval or settings.rewrite.verify_before_retrieval

        if prompts_path is None:
            prompts_path = Path(__file__).parent / "prompts" / "rewriter_prompts.yaml"

        self.prompts = self._load_prompts(prompts_path)
        self.analyzer = analyzer or LinguisticAnalyzer()

        cache_key = "adaptive_rewriter_llm"

        def _create_llm():
            return LLMInference(
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
            )

        # is this the right way? maybe used cached model from generation? but its before its init...
        self.llm = llm or model_registry.get_or_create(cache_key, _create_llm)

        logger.info(
            f"AdaptiveQueryRewriter initialized "
            f"(verify={verify_before_retrieval}, enabled={enabled})"
        )

    def _load_prompts(self, path: Path) -> Dict[str, str]:
        """Load prompts from YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Prompts file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        logger.info(f"Loaded prompts from {path}")
        return prompts

    def rewrite(self, query: str) -> Dict[str, Any]:
        """Rewrite query with linguistic context.

        Args:
            query: Original query

        Returns:
            Dict with:
              - original: str
              - queries: List[str] (may be [original] or [sub1, sub2, ...])
              - is_multihop: bool
              - reasoning: str
              - linguistic_features: LinguisticFeatures
        """
        if not self.enabled:
            return {
                "original": query,
                "queries": [query],
                "is_multihop": False,
                "reasoning": "rewriting disabled",
                "linguistic_features": None,
            }

        # Step 1: Linguistic analysis
        features = self.analyzer.analyze(query)
        logger.debug(f"Linguistic features: {features.num_tokens} tokens, {features.num_clauses} clauses")

        # Step 2: LLM Unified decision (decision + decompose/expand in one call)
        result = self._make_decision(query, features)

        if not result["is_multihop"]:
            if result["action"] == "expand" and result.get("expanded_query"):
                logger.info(f"Expanded: '{query}' → '{result['expanded_query']}'")
                return {
                    "original": query,
                    "queries": [result["expanded_query"]],
                    "is_multihop": False,
                    "reasoning": result["reasoning"],
                    "linguistic_features": features,
                }

            # passthrough
            logger.info(f"No rewriting needed, passing through, reason: {result['reasoning']}")
            return {
                "original": query,
                "queries": [query],
                "is_multihop": False,
                "reasoning": result["reasoning"],
                "linguistic_features": features,
            }

        # Step 3: Multihop = true, sub-queries in given result
        sub_queries = result.get("sub_queries", [query])
        if not sub_queries:
            sub_queries = [query]

        logger.info(f"Query {query} decomposed into sub-queries: {sub_queries}")

        # # Step 4: Verify before retrival
        # if self.verify_before_retrieval and len(sub_queries) > 1:
        #     verified = self._verify_subqueries(query, sub_queries)
        #     if not verified["valid"]:
        #         logger.warning(f"Verification failed: {verified['issues']}")
        #         if verified["corrected_queries"]:
        #             sub_queries = verified["corrected_queries"]
        #         else:
        #             return {
        #                 "original": query,
        #                 "queries": [query],
        #                 "is_multihop": False,
        #                 "reasoning": f"verification failed: {verified['issues']}",
        #                 "linguistic_features": features,
        #             }

        return {
            "original": query,
            "queries": sub_queries,
            "is_multihop": True,
            "reasoning": result["reasoning"],
            "linguistic_features": features,
        }


    def _make_decision(
            self,
            query: str,
            features: LinguisticFeatures
    ) -> Dict[str, Any]:
        """Single LLM call for decision + decomposition/expansion."""
        prompt_config = self.prompts["unified_decision"]
        system = prompt_config["system"]
        template = prompt_config["template"]

        prompt = f"{system}\n\n{template}".format(
            query=query,
            linguistic_context=features.to_context_string(),
        )

        logger.info(f"LLM decision prompt: {prompt}")


        # TODO: Add fallback with retry - if LLM returns non-JSON, retry once before falling back to basic passthrough
        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                chain_of_thought_enabled=False,
            ).strip()

            result = safe_parse(response)
            if result:
                return result

        except Exception as e:
            logger.error(f"Decision LLM error: {e}")

        return {
            "is_multihop": False,
            "action": "passthrough",
            "sub_queries": None,
            "expanded_query": None,
            "reasoning": "decision LLM error",
            "confidence": 0.0,
        }

    # currently not used
    def _verify_subqueries(self, original: str, sub_queries: List[str]) -> Dict[str, Any]:
        """Verify sub-queries before retrieval."""
        prompt_config = self.prompts["verification"]
        system = prompt_config["system"]
        template = prompt_config["template"]

        prompt = f"{system}\n\n{template}".format(
            original=original,
            sub_queries=json.dumps(sub_queries, ensure_ascii=False),
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                chain_of_thought_enabled=False,
            ).strip()

            parsed = safe_parse(response)
            if parsed:
                return parsed

        except Exception as e:
            logger.error(f"Verification failed: {e}")

        # Fallback: assume valid
        return {"valid": True, "issues": [], "corrected_queries": None}

# old code for refactor -> save for if the current implementation is not enough

# def _decompose(
#         self,
#         query: str,
#         features: LinguisticFeatures
# ) -> List[str]:
#     """Decompose query into sub-queries based on linguistic context."""
#     prompt_config = self.prompts["decompose"]
#     system = prompt_config["system"]
#     template = prompt_config["template"]
#
#     prompt = f"{system}\n\n{template}".format(
#         query=query,
#         linguistic_context=features.to_context_string(),
#     )
#
#     logger.debug(f"Decompose prompt length: {len(prompt)} chars")
#
#     try:
#         response = self.llm.generate(
#             prompt=prompt,
#             temperature=self.temperature,
#             max_new_tokens=self.max_tokens,
#             chain_of_thought_enabled=False,
#         ).strip()
#
#         logger.debug(f"Decompose raw response: {response[:300]}")
#
#         parsed = safe_parse(response)
#
#         # Handle multiple possible keys: sub_queries, sub_questions, queries, questions
#         if isinstance(parsed, dict):
#             for key in ["sub_queries", "sub_questions", "queries", "questions"]:
#                 if key in parsed:
#                     sub_queries = parsed[key]
#                     break
#             else:
#                 logger.warning(f"No valid sub-queries key found in: {list(parsed.keys())}")
#                 return [query]
#         elif isinstance(parsed, list):
#             sub_queries = parsed
#             logger.info("LLM returned list directly (not wrapped in object)")
#         else:
#             logger.warning(f"Unexpected parsed format: {type(parsed)}")
#             return [query]
#
#         # Validate sub-queries
#         if not isinstance(sub_queries, list) or len(sub_queries) == 0:
#             logger.warning(f"Invalid sub_queries: {sub_queries}")
#             return [query]
#         valid_queries = [q for q in sub_queries if isinstance(q, str) and len(q.strip()) > 0]
#
#         if not valid_queries:
#             logger.warning("No valid sub-queries after filtering")
#             return [query]
#
#         logger.info(f"Successfully decomposed into {len(valid_queries)} sub-queries")
#         return valid_queries
#
#     except Exception as e:
#         logger.error(f"Decompose LLM error: {e}")
#
#     return [query]
#
#
# def _expand_simple(
#         self,
#         query: str,
#         features: LinguisticFeatures
# ) -> str:
#     """Expand simple query based on linguistic context."""
#     prompt_config = self.prompts["expand"]
#     system = prompt_config["system"]
#     template = prompt_config["template"]
#
#     prompt = f"{system}\n\n{template}".format(
#         query=query,
#         linguistic_context=features.to_context_string(),
#     )
#
#     try:
#         response = self.llm.generate(
#             prompt=prompt,
#             temperature=self.temperature,
#             max_new_tokens=self.max_tokens,
#             chain_of_thought_enabled=False,
#         ).strip()
#
#         parsed = self._safe_parse(response)
#         if parsed and "expanded_query" in parsed:
#             return parsed["expanded_query"]
#
#     except Exception as e:
#         logger.error(f"Expand LLM error: {e}")
#
#     return query
