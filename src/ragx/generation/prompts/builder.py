from __future__ import annotations

import logging
from collections import defaultdict

import yaml
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from langdetect import detect, LangDetectException

from src.ragx.generation.prompts.utils.prompt_config import PromptConfig
from src.ragx.generation.prompts.utils.prompt_tools import get_confidence_level, get_quality_check_reminder, \
    format_contexts_simple, get_default_template
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Advanced prompt builder with template support and smart formatting."""

    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir or (Path(__file__).parent / "templates")
        self._templates_cache = {}
        self._components_cache = {}

    def detect_language(self, text: str) -> str:
        """Detect language of the text. Currently supports Polish and English language detection."""
        try:
            lang_code = detect(text)
            return "Polish" if lang_code == "pl" else "English"
        except LangDetectException:
            if re.search(r'[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', text):
                return "Polish"
            else:
                return "English"

    def format_contexts_advanced(
            self,
            contexts: List[Dict[str, Any]],
            include_metadata: bool = True,
            group_by_doc: bool = False
    ) -> str:
        """
        Advanced context formatting with metadata and grouping options.

        Args:
            contexts: List of context dictionaries
            include_metadata: Include scores and positions
            group_by_doc: Group chunks from same document
        """
        if not contexts:
            return "[No sources available]"

        if group_by_doc:
            return self._format_grouped_contexts(contexts, include_metadata)

        formatted = []
        for idx, ctx in enumerate(contexts, 1):
            text = ctx.get("text", "").strip()
            title = ctx.get("doc_title", "Unknown")
            position = ctx.get("position", 0)
            total = ctx.get("total_chunks", 1)

            if total > 1:
                parts = [f"[{idx}] SOURCE: {title} (Fragment {position + 1}/{total})"]
            else:
                parts = [f"[{idx}] SOURCE: {title}"]

            if include_metadata:
                metadata_parts = []

                rerank_score = ctx.get("rerank_score") if ctx.get("rerank_score") is not None else ctx.get(
                    "retrieval_score")
                if rerank_score is not None:
                    confidence = get_confidence_level(rerank_score)

                    metadata_parts.append(f"relevance: {rerank_score:.3f} ({confidence})")

                elif ctx.get("retrieval_score") is not None:
                    score = ctx.get("retrieval_score")
                    metadata_parts.append(f"similarity: {score:.3f}")

                position = ctx.get("position", 0)
                total = ctx.get("total_chunks", 1)
                if total > 1:
                    metadata_parts.append(f"chunk: {position}/{total}")

                if metadata_parts:
                    parts.append(f"[{', '.join(metadata_parts)}]")

            parts.append(f"CONTENT: {text}")

            if include_metadata and ctx.get("rerank_score", 1.0) < 0.3:
                parts.append("Note: Low relevance score - use with caution")

            formatted.append("\n".join(parts))

        return "\n" + "=" * 50 + "\n" + "\n".join(formatted) + "\n" + "=" * 50

    def _format_grouped_contexts(
            self,
            contexts: List[Dict[str, Any]],
            include_metadata: bool = True
    ) -> str:
        """Format contexts grouped by document."""
        grouped = {}
        for idx, ctx in enumerate(contexts, 1):
            title = ctx.get("doc_title", "Unknown")
            if title not in grouped:
                grouped[title] = []
            ctx["_idx"] = idx
            grouped[title].append(ctx)

        formatted = []
        for title, chunks in grouped.items():
            doc_parts = [f"DOCUMENT: {title}"]

            if include_metadata:
                scores = [c.get("rerank_score", c.get("retrieval_score", 0.0)) for c in chunks]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                doc_parts.append(f"[Average relevance: {avg_score:.3f}, Chunks: {len(chunks)}]")

            # if empty line
            doc_parts.append("")

            for chunk in sorted(chunks, key=lambda c: c.get("position", 0)):
                idx = chunk['_idx']
                text = chunk.get("text", "").strip()
                doc_parts.append(f"[{idx}] {text}")
                doc_parts.append("")  # format purpose (separation of chunls)

            formatted.append("\n".join(doc_parts))

        return "\n" + "=" * 50 + "\n" + "\n\n".join(formatted) + "\n" + "=" * 50

    def build(
            self,
            query: str,
            contexts: List[Dict[str, Any]],
            template_name: str = 'basic',
            chat_history: Optional[List[Dict[str, str]]] = None,
            max_history: Optional[int] = None,
            config: Optional[PromptConfig] = None,
            is_multihop: bool = False,
            sub_queries: Optional[List[str]] = None,
            **kwargs
    ) -> Union[str, Tuple[str, Dict[int, str]]]:
        """
        Build advanced prompt from template.

        Args:
            query: User question
            contexts: Retrieved contexts with metadata
            template_name: Template to use
            chat_history: Optional conversation history
            max_history: Maximum history entries
            config: Prompt configuration options
            is_multihop: Whether this is a multihop query
            sub_queries: Decomposed sub-queries (for multihop)
            **kwargs: Additional template variables
        """
        config = config or PromptConfig()

        if is_multihop and sub_queries:
            return self.build_multihop(
                original_query=query,
                sub_queries=sub_queries,
                contexts=contexts,
                config=config,
                **kwargs
            )

        template_data = self.load_template(template_name)
        template_str = template_data.get("template", "")

        detected_language = self.detect_language(query) if config.detect_language else "English"

        language_instruction = f"Answer entirely in {detected_language}"
        if detected_language == "Polish":
            language_instruction = "Odpowiedz WYŁĄCZNIE po polsku"

        if template_name == 'enhanced':
            contexts_formatted = self.format_contexts_advanced(
                contexts,
                include_metadata=config.include_metadata,
                # group_by_doc=query
            )
        else:
            contexts_formatted = format_contexts_simple(contexts)

        # model based, change second one when needed!
        think_tag_start = ""
        think_tag_end = ""
        if config.use_cot and config.think_tag_style == "qwen":
            think_tag_start = "<think>"
            think_tag_end = "</think>"
        # Most other models use the <thinking> tag (e.g., LLaMA), so this style is multipurpose
        elif config.use_cot and config.think_tag_style == "llama":
            think_tag_start = "<thinking>"
            think_tag_end = "</thinking>"

        history_str = ""
        if chat_history:
            history_str = self._format_chat_history(chat_history, max_history)

        prompt_vars = {
            "query": query,
            "contexts": contexts_formatted,
            "contexts_formatted": contexts_formatted,
            "contexts_with_metadata": contexts_formatted,  # enhanced purposes
            "chat_history": history_str,
            "detected_language": detected_language,
            "language_instruction": language_instruction,
            "think_tag_start": think_tag_start,
            "think_tag_end": think_tag_end,
            **kwargs
        }

        try:
            prompt = template_str.format(**prompt_vars)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            prompt = self._build_fallback_prompt(query, contexts, detected_language)

        if template_name == 'enhanced' and config.strict_citations:
            prompt += get_quality_check_reminder(detected_language)

        logger.info(f"Prompt built: {prompt}")
        return prompt.strip()

    def build_multihop(
            self,
            original_query: str,
            sub_queries: List[str],
            contexts: List[Dict[str, Any]],
            config: Optional[PromptConfig] = None,
            **kwargs
    ) -> Tuple[str, Dict[int, str]]:
        """Build multihop prompt with grouped contexts.

        Args:
            original_query: Original complex query
            sub_queries: Decomposed sub-queries
            contexts: Retrieved contexts (may have 'source_subquery' field)
            config: Prompt configuration
            **kwargs: Additional template variables

        Returns:
            Formatted prompt
        """
        config = config or PromptConfig(
            use_cot=True,
            include_metadata=True,
            strict_citations=True,
            detect_language=True,
        )

        # Group contexts by source sub-query
        grouped = self._group_context_by_subquery(contexts)
        grouped_contexts_str, citation_mapping = self._format_grouped_contexts_for_multihop(
            grouped, sub_queries
        )

        # Format sub-queries list
        sub_queries_list = "\n".join(
            [f"{i + 1}. {sq}" for i, sq in enumerate(sub_queries)]
        )

        detected_language = self.detect_language(original_query)
        think_tag_start = ""
        think_tag_end = ""
        if config.use_cot and config.think_tag_style == "qwen":
            think_tag_start = "<think>"
            think_tag_end = "</think>"
        elif config.use_cot and config.think_tag_style == "llama":
            think_tag_start = "<thinking>"
            think_tag_end = "</thinking>"

        template_data = self.load_template("multihop")
        template_str = template_data.get("template", "")

        prompt = template_str.format(
            original_query=original_query,
            sub_queries_list=sub_queries_list,
            grouped_contexts=grouped_contexts_str,
            detected_language=detected_language,
            think_tag_start=think_tag_start,
            think_tag_end=think_tag_end,
            **kwargs
        )

        logger.info(f"Multihop prompt built for query: {prompt}")
        return prompt.strip(), citation_mapping

    def _group_context_by_subquery(
            self,
            contexts: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group contexts by source sub-query."""
        grouped = defaultdict(list)
        for ctx in contexts:
            # Check if context has fusion metadata with multiple source sub-queries
            fusion_meta = ctx.get("fusion_metadata", {}) or {}
            source_subqueries = fusion_meta.get("source_subqueries", [])

            if source_subqueries:
                # Assign context to ALL relevant queries
                for subquery in source_subqueries:
                    grouped[subquery].append(ctx)
            else:
                source_subquery = ctx.get("source_subquery", "")
                grouped[source_subquery].append(ctx)

        return dict(grouped)

    def _format_grouped_contexts_for_multihop(
            self,
            grouped: Dict[str, List[Dict[str, Any]]],
            sub_queries: List[str],
    ) -> Tuple[str, Dict[int, str]]:
        """Format grouped contexts for multihop prompt."""
        if not grouped:
            return "[No contexts available]", {}

        formatted_sections = []
        global_idx = 1
        citation_to_doc_id = {}

        # Iterate in sub-query order
        for sub_query in sub_queries:
            contexts = grouped.get(sub_query, [])

            if not contexts:
                section = [
                    f"\n{'=' * 60}",
                    f"SUB-QUERY: {sub_query}",
                    f"{'=' * 60}",
                    "[No contexts found for this sub-query]",
                    ""
                ]
                formatted_sections.append("\n".join(section))
                continue

            section = [
                f"\n{'=' * 60}",
                f"SUB-QUERY: {sub_query}",
                f"{'=' * 60}\n"
            ]

            for ctx in contexts:
                text = ctx.get("text", "").strip()
                title = ctx.get("doc_title", "Unknown")
                position = ctx.get("position", 0)
                total = ctx.get("total_chunks", 1)

                real_doc_id = ctx.get("id") or str(global_idx)
                citation_to_doc_id[global_idx] = real_doc_id

                if total > 1:
                    section.append(f"[{global_idx}] SOURCE: {title} (Fragment {position + 1}/{total})")
                else:
                    section.append(f"[{global_idx}] SOURCE: {title}")

                # Scores
                final_score = float(ctx.get("final_score"))
                if final_score is not None:
                    confidence = get_confidence_level(final_score)
                    section.append(f"    Relevance: {final_score:.3f} ({confidence})")

                if final_score is not None and final_score < 0.35:
                    section.append(f"    BE WARY, THIS SOURCE MAY NOT BE RELEVANT!")

                fusion_meta = ctx.get("fusion_metadata", {}) or {}
                if fusion_meta.get("num_occurrences", 1) > 1:
                    source_sqs = fusion_meta.get("source_subqueries", [])
                    section.append(
                        f"    ℹ Also relevant to {len(source_sqs)} sub-queries"
                    )

                section.append(f"    CONTENT: {text}\n")
                global_idx += 1

            formatted_sections.append("\n".join(section))

        return "\n".join(formatted_sections), citation_to_doc_id

    def _format_chat_history(
            self,
            chat_history: Optional[List[Dict[str, str]]] = None,
            max_history: Optional[int] = None
    ) -> str:
        """Format chat history for a prompt.

        Args:
            chat_history: List of chat messages with 'role' and 'content'
            max_history: Maximum number of recent messages to include
        """
        if not chat_history:
            return ""

        max_history = max_history if max_history is not None else settings.chat.max_history

        if max_history is not None and len(chat_history) > max_history:
            # if the max history is exceeded, take the most recent messages - last in the list is most recent
            recent = chat_history[-max_history:]
        else:
            recent = chat_history

        formatted = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def _build_fallback_prompt(
            self,
            query: str,
            contexts: List[Dict[str, Any]],
            language: str
    ) -> str:
        """Fallback prompt if template fails."""
        contexts_str = format_contexts_simple(contexts)
        return f"""
            Sources:
            {contexts_str}
            
            Question: {query}
            
            Instructions: Answer only from sources with [N] citations. Language: {language}
            
            Answer:
        """

    def load_template(
            self,
            name: str
    ) -> dict:
        """Load and cache template data."""
        if name in self._templates_cache:
            return self._templates_cache[name]

        template_path = self.template_dir / f"{name}.yaml"
        if not template_path.exists():
            logger.warning(f"Template not found: {template_path}")
            return {"template": get_default_template()}

        with open(template_path, 'r', encoding='utf-8') as f:
            template = yaml.safe_load(f)

        self._templates_cache[name] = template
        return template


def build_prompt_for_pipeline(
        pipeline_type: str,
        query: str,
        contexts: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        **kwargs
) -> Tuple[str, PromptConfig]:
    """
    Build optimal prompt for pipeline type.

    Args:
        pipeline_type: Pipeline type
        query: User question
        contexts: Retrieved contexts with metadata
        model_name: Model name

    Returns:
        Tuple of (prompt, config_used)
    """
    builder = PromptBuilder()

    if pipeline_type == "baseline":
        template = "basic"
        config = PromptConfig(
            use_cot=False,
            include_metadata=False,
            strict_citations=True,
            detect_language=True,
            check_contradictions=False,
            confidence_scoring=False,
            think_tag_style="none"
        )
    else:
        template = "enhanced"
        config = PromptConfig(
            use_cot=True,
            include_metadata=True,
            strict_citations=True,
            detect_language=True,
            check_contradictions=True,
            confidence_scoring=True,
            think_tag_style="qwen" if model_name and "qwen" in model_name.lower() else "none"
        )

    prompt = builder.build(
        query=query,
        contexts=contexts,
        template_name=template,
        config=config,
        **kwargs
    )

    return prompt, config
