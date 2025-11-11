from __future__ import annotations

import logging
from typing import Optional, Iterator

import ollama

from src.ragx.utils.settings import settings
from src.ragx.generation.providers.helpers.ollama_startup import ensure_ollama_running

logger = logging.getLogger(__name__)


class OllamaProvider:
    """Ollama-based LLM provider."""

    def __init__(
            self,
            model_name: Optional[str] = None,
            host: Optional[str] = None,
            timeout: int = 30,
    ):
        """
        Initialize Ollama client.

        Args:
            model_name: Ollama model name (format: model:tag)
            host: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model_name = model_name or settings.llm.model_name
        self.host = host or settings.llm.ollama_host
        self.timeout = timeout

        logger.info(f"Loading Ollama model: {self.model_name}")
        logger.info(f"  - Host: {self.host}")
        logger.info(f"  - Timeout: {self.timeout} seconds")

        try:
            if not ensure_ollama_running(timeout=self.timeout, host=self.host):
                raise RuntimeError("Ollama server not running")
            models = ollama.list()
            available = [m.model for m in models.models]
            logger.info(f"Available Ollama models: {available}")

            if self.model_name not in available:
                logger.warning(f"Model {self.model_name} not found in Ollama models. Using default model.")
                logger.info(f"Pulling model {self.model_name} from Ollama server. (This may take a while...)")
                ollama.pull(self.model_name)
                logger.info(f"Model {self.model_name} pulled from Ollama server.")
        except Exception as e:
            logger.error(f"Failed to load Ollama model: {e}")
            raise RuntimeError("Ollama model load failed") from e

        logger.info(f"Loaded Ollama model: {self.model_name}")

    def generate(
            self,
            prompt: str,
            temperature: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            chain_of_thought_enabled: Optional[bool] = True,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            chain_of_thought_enabled: Enable Chain of Thought (CoT)

        Returns:
            Generated text
        """
        chain_of_thought_enabled = chain_of_thought_enabled or False
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "temperature": temperature or settings.llm.temperature,
                "num_predict": max_new_tokens or settings.llm.max_new_tokens,
                "top_p": top_p or settings.llm.top_p,
                "think": chain_of_thought_enabled,
                "repeat_penalty": 1.5,
            }
        )
        # logger.info(f"Response: {response}")

        generated_text = response.get('response', '')
        logger.info(f"Generated text: {generated_text}")

        if chain_of_thought_enabled:
            thinking_process = response.get('thinking', '')
            logger.info(f"Thinking process: {thinking_process}")

        if not generated_text or len(generated_text.strip()) == 0:
            logger.error("Error response from Ollama! Common error with lack of max_new_tokens.")
            return "I apologise, I couldn't generate a proper response. "

        return generated_text

    def generate_stream(
            self,
            prompt: str,
            temperature: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter

        Yields:
            Generated text chunks
        """
        stream = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            stream=True,
            options={
                "temperature": temperature or settings.llm.temperature,
                "num_predict": max_new_tokens or settings.llm.max_new_tokens,
                "top_p": top_p or settings.llm.top_p,
            }
        )

        for chunk in stream:
            if 'response' in chunk:
                yield chunk['response']

    def chat_generate(
            self,
            messages: list[dict],
            temperature: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate using chat format (better for multi-turn conversations).

        Args:
            messages: Chat messages [{"role": "user", "content": "..."}]
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature": temperature or settings.llm.temperature,
                "num_predict": max_new_tokens or settings.llm.max_new_tokens,
            }
        )

        return response['message']['content']
