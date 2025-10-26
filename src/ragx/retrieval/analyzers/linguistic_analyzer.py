from __future__ import annotations

import logging
import spacy
from spacy.tokens import Doc, Token

from src.ragx.retrieval.analyzers.linguistic_features import LinguisticFeatures
from src.ragx.utils.model_registry import model_registry

logger = logging.getLogger(__name__)


class LinguisticAnalyzer:
    """Extract linguistic features using spaCy (language-agnostic)."""

    def __init__(
            self,
            model_name: str = "pl_core_news_md",
            fallback_model: str = "en_core_web_sm",
    ):
        """
        Initialize linguistic analyzer.

        Args:
            model_name: Name of spaCy model to use
            fallback_model: Name of spaCy model to use if model_name is not available
        """
        cache_key = f"spacy:{model_name}"

        def _load_spacy():
            try:
                logger.info(f"Loading spaCy model: {model_name}")
                return spacy.load(model_name)
            except OSError:
                logger.warning(f"Model {model_name} not found, using fallback model: {fallback_model}")
                try:
                    return spacy.load(fallback_model)
                except OSError:
                    logger.error(f"Fallback model {fallback_model} not found. Please install it.")
                    raise ImportError(
                        f"Install spaCy models:\n"
                        f"  python -m spacy download {model_name}\n"
                        f"  python -m spacy download {fallback_model}"
                    )

        self.nlp = model_registry.get_or_create(cache_key, _load_spacy)
        logger.info(f"LinguisticAnalyzer ready: {self.nlp.meta['name']}")

    def analyze(self, query: str) -> LinguisticFeatures:
        """Extract linguistic features from a query using spaCy."""

        if not query or len(query.strip()) < 2:
            return self._empty_features(query)

        doc: Doc = self.nlp(query)

        pos_sequence = [token.pos_ for token in doc]

        # First 20 for understending the context -> increase once i get this to work TODO
        dep_tree = [
            (token.dep_, token.head.text, token.text)
            for token in doc
        ][:20]

        # description of the entity
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # counts the sentences that have a clause modifier
        num_clauses = len(list(doc.sents))
        for token in doc:
            # relcl: Relative clause modifier (e.g., "who runs fast" in "The man who runs fast").
            # ccomp: Clausal complement (e.g., "that he left" in "I know that he left").
            # advcl: Adverbial clause modifier (e.g., "when he arrived" in "He left when he arrived").
            # acl: Clausal modifier of a noun (e.g., "to read" in "a book to read").
            if token.dep_ in ("relcl", "ccomp", "advcl", "acl"):
                num_clauses += 1

        syntax_depth = self._calculate_depth(doc)

        has_relative_clauses = any(token.dep_ == "relcl" for token in doc)

        has_conjunctions = any(token.pos_ in ("CCONJ", "SCONJ") for token in doc)

        return LinguisticFeatures(
            query=query,
            pos_sequence=pos_sequence,
            dep_tree=dep_tree,
            entities=entities,
            num_tokens=len(doc),
            num_clauses=num_clauses,
            syntax_depth=syntax_depth,
            has_relative_clauses=has_relative_clauses,
            has_conjunctions=has_conjunctions,
        )

    def _calculate_depth(self, doc: Doc) -> int:
        """Calculate the depth of the syntax tree."""

        def token_depth(token: Token) -> int:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
                if depth > 200:  # safeguard
                    break
            return depth

        return max((token_depth(token) for token in doc), default=0)

    def _empty_features(self, query: str) -> LinguisticFeatures:
        """Return empty features for a query."""
        return LinguisticFeatures(
            query=query,
            pos_sequence=[],
            dep_tree=[],
            entities=[],
            num_tokens=0,
            num_clauses=0,
            syntax_depth=0,
            has_relative_clauses=False,
            has_conjunctions=False,
        )
