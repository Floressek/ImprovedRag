from __future__ import annotations

import logging
import json

import yaml
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.ragx.generation.inference import LLMInference
from src.ragx.retrieval.analyzers.linguistic_analyzer import LinguisticAnalyzer, LinguisticFeatures
from src.ragx.retrieval.rewriters.tools.parse import (
    safe_parse,
    JSONValidator,
    RetryConfig,
    validate_rewriter_response,
    validate_verification_response,
)
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
        self.max_tokens = max_tokens if max_tokens is not None else settings.rewrite.max_tokens
        self.temperature = temperature if temperature is not None else settings.rewrite.temperature
        self.enabled = enabled if enabled is not None else settings.rewrite.enabled
        self.verify_before_retrieval = verify_before_retrieval if verify_before_retrieval is not None else settings.rewrite.verify_before_retrieval

        if prompts_path is None:
            prompts_path = Path(__file__).parent / "prompts" / "rewriter_prompts.yaml"

        self.prompts = self._load_prompts(prompts_path)
        self.analyzer = analyzer or LinguisticAnalyzer()
        self.json_validator = JSONValidator(RetryConfig(max_retries=2, initial_delay=0.3))

        cache_key = "adaptive_rewriter_llm"

        def _create_llm():
            return LLMInference(
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
            )

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

        query_type = result.get("query_type", "general")

        if not result["is_multihop"]:
            if result["action"] == "expand" and result.get("expanded_query"):
                logger.info(f"Expanded: '{query}' → '{result['expanded_query']}'")
                return {
                    "original": query,
                    "queries": [result["expanded_query"]],
                    "is_multihop": False,
                    "reasoning": result["reasoning"],
                    "query_type": query_type,
                    "linguistic_features": features,
                }

            # passthrough
            logger.info(f"No rewriting needed, passing through, reason: {result['reasoning']}")
            return {
                "original": query,
                "queries": [query],
                "is_multihop": False,
                "reasoning": result["reasoning"],
                "query_type": query_type,
                "linguistic_features": features,
            }

        # Step 3: Multihop = true, sub-queries in given result
        sub_queries = result.get("sub_queries", [query])
        if not sub_queries:
            sub_queries = [query]

        logger.info(f"Query {query} decomposed into sub-queries: {sub_queries}")

        # # Step 4: Verify before retrival (to be fixed in updated version)
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
            "query_type": query_type,
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

        # Generator function for retry logic
        def generate_response() -> str:
            return self.llm.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                chain_of_thought_enabled=True,
            ).strip()

        # Use validator with retry
        success, result, metadata = self.json_validator.validate_with_retry(
            generator_func=generate_response,
            parse_func=safe_parse,
            validator_func=validate_rewriter_response,
        )

        if success and result:
            logger.info(
                f"Decision parsed successfully after {metadata['attempts']} attempt(s) "
                f"in {metadata['total_time']:.2f}s"
            )
            return result

        # Fallback after all retries failed
        logger.error(
            f"All {metadata['attempts']} attempts failed. "
            f"Errors: {metadata['errors']}"
        )
        return {
            "is_multihop": False,
            "action": "passthrough",
            "sub_queries": None,
            "expanded_query": None,
            "reasoning": "LLM failed to produce valid JSON after retries",
            "confidence": 0.0,
        }

    # currently not used -> to be implemented
    def _verify_subqueries(self, original: str, sub_queries: List[str]) -> Dict[str, Any]:
        """Verify sub-queries before retrieval."""
        prompt_config = self.prompts["verification"]
        system = prompt_config["system"]
        template = prompt_config["template"]

        prompt = f"{system}\n\n{template}".format(
            original=original,
            sub_queries=json.dumps(sub_queries, ensure_ascii=False),
        )

        # Generator function for retry logic
        def generate_response() -> str:
            return self.llm.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                chain_of_thought_enabled=True,
            ).strip()

        # Use validator with retry
        success, result, metadata = self.json_validator.validate_with_retry(
            generator_func=generate_response,
            parse_func=safe_parse,
            validator_func=validate_verification_response,
        )

        if success and result:
            logger.info(
                f"Verification parsed successfully after {metadata['attempts']} attempt(s) "
                f"in {metadata['total_time']:.2f}s"
            )
            return result

        # Fallback: assume valid if parsing failed
        logger.warning(
            f"Verification parsing failed after {metadata['attempts']} attempts. "
            f"Assuming sub-queries are valid."
        )
        return {"valid": True, "issues": [], "corrected_queries": None}

