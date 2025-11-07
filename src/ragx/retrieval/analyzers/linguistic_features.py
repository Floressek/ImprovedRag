from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LinguisticFeatures:
    """
    Linguistic features extracted from a text document.

    Attributes:
        query: The original query string.
        pos_sequence: POS sequence of tokens.
        dep_tree: Dependency tree of tokens.
        entities: List of named entities.
        num_tokens: Number of tokens in the document.
        num_clauses: Number of clauses in the document.
        syntax_depth: Depth of the syntax tree.
        has_relative_clauses: Whether the document contains relative clauses.
        has_conjunctions: Whether the document contains conjunctions.
    """
    query: str
    pos_sequence: List[str]  # ["NOUN", "PROPN", ...]
    dep_tree: List[Tuple[str, str, str]]  # [(dep, head, child, ...)]
    entities: List[Tuple[str, str]]  # [(text, label)]
    num_tokens: int
    num_clauses: int
    syntax_depth: int
    has_relative_clauses: bool
    has_conjunctions: bool

    def to_context_string(self) -> str:
        """Format features as context for LLM prompt."""
        ent_str = ", ".join([f"{text} ({label})" for text, label in self.entities])
        dep_str = "; ".join([f"{dep}({head}→{child})" for dep, head, child in self.dep_tree[:10]])

        return f"""
        Linguistic Analysis:
                - Tokens: {self.num_tokens}
                - POS sequence: {' '.join(self.pos_sequence)}
                - Entities: {ent_str if ent_str else 'none'}
                - Dependencies: {dep_str}
                - Clauses: {self.num_clauses}
                - Syntax depth: {self.syntax_depth}
                - Has relative clauses: {self.has_relative_clauses}
                - Has conjunctions: {self.has_conjunctions}
                """
