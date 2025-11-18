from __future__ import annotations

import logging
from typing import List, Optional, Any

from langchain_core.language_models.llms import LLM
from langchain_core.embeddings import Embeddings
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from src.ragx.generation.inference import LLMInference
from src.ragx.retrieval.embedder.embedder import Embedder

logger = logging.getLogger(__name__)


class LLMInferenceAdapter(LLM):
    """LangChain LLM adapter for our LLMInference class.

    This allows RAGAS to use our multi-provider LLM infrastructure
    (API, Ollama, HuggingFace) instead of being locked to OpenAI.
    """

    llm_inference: LLMInference
    temperature: float = 0.5
    max_tokens: int = 4092

    def __init__(
            self,
            provider: str = "api",
            temperature: float = 0.5,
            max_tokens: int = 4092,
            **kwargs: Any,
    ):
        """Initialize adapter with LLMInference instance.

        Args:
            provider: LLM provider ("api", "ollama", "huggingface")
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional LangChain LLM kwargs
        """
        llm_inference = LLMInference(
            provider=provider,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        super().__init__(llm_inference=llm_inference, **kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Initialized LLMInferenceAdapter with provider: {provider}")

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "llm_inference"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Call our LLMInference.generate() method.

        Args:
            prompt: Input prompt
            stop: Stop sequences (not used by our implementation)
            run_manager: LangChain callback manager
            **kwargs: Additional generation kwargs

        Returns:
            Generated text
        """
        response = self.llm_inference.generate(
            prompt=prompt,
            temperature=kwargs.get("temperature", self.temperature),
            max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
            chain_of_thought_enabled=False,  # RAGAS doesn't need CoT
        )
        return response


class EmbedderAdapter(Embeddings):
    """LangChain Embeddings adapter for our Embedder class.

    This allows RAGAS to use our local SentenceTransformer embeddings
    instead of requiring OpenAI embeddings API.
    """

    embedder: Embedder

    def __init__(
            self,
            model_id: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: Any,
    ):
        """Initialize adapter with Embedder instance.

        Args:
            model_id: SentenceTransformers model ID
            device: Device to run on ("cpu", "cuda", "auto")
            **kwargs: Additional kwargs
        """
        super().__init__(**kwargs)
        self.embedder = Embedder(
            model_id=model_id,
            device=device,
            show_progress=False,  # RAGAS doesn't need progress bars
        )
        logger.info(f"Initialized EmbedderAdapter with model: {self.embedder.model_id}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        # Our Embedder.embed_texts() returns list[list[float]] when convert_to_numpy=False
        embeddings = self.embedder.embed_texts(
            texts=texts,
            convert_to_numpy=False,  # Get list of lists, not numpy array
            add_prefix=True,  # Treat as passages/documents
        )
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding as list of floats
        """
        # Our Embedder.embed_query() already returns list[float]
        embedding = self.embedder.embed_query(text)
        return embedding