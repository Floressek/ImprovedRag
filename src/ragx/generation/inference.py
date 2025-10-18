from __future__ import annotations

import logging
from typing import Optional, Iterator

import torch
from transformers import TextIteratorStreamer
from threading import Thread

from src.ragx.generation.model import LLMModel
from src.ragx.utils.model_registry import model_registry
from src.ragx.utils.settings import settings
from src.ragx.generation.providers.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class LLMInference:
    """LLM inference with multi-provider support (HF Transformers, Ollama, vLLM).

    Provider selection priority:
    1. Explicitly passed provider parameter
    2. Settings (LLM_PROVIDER in .env)
    3. Default to 'huggingface'

    All providers are cached via model_registry for efficient reuse.
    """

    def __init__(
            self,
            llm_model: Optional[LLMModel] = None,
            temperature: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
            provider: Optional[str] = None,
    ):
        self.temperature = temperature if temperature is not None else settings.llm.temperature
        self.max_new_tokens = max_new_tokens or settings.llm.max_new_tokens
        self.provider = provider or settings.llm.provider

        logger.info(f" 🚀 Initializing LLMInference with provider: {self.provider}")
        if llm_model is not None:
            self.provider = 'huggingface'
            self.llm_model = llm_model
            self.tokenizer = llm_model.get_tokenizer()
            self.model = llm_model.get_model()
            self._provider_instance = None
            logger.info(f"✓ Using passed LLMModel: {llm_model.model_id}")
            return

        self.model_id = settings.llm.model_id

        cache_key = f"llm_provider:{self.provider}:{self.model_id}"

        def _create_provider():
            """Factory function for model_registry"""
            logger.info(f" 📦 Creating LLM provider instance: {self.provider}")
            return self._initialize_provider()

        # using models registry to cache model instances
        self._provider_instance = model_registry.get_or_create(
            cache_key,
            _create_provider
        )

        # for HuggingFace
        if self.provider == 'huggingface':
            self.llm_model = self._provider_instance
            self.tokenizer = self.llm_model.get_tokenizer()
            self.model = self.llm_model.get_model()
        else:
            self.llm_model = None
            self.tokenizer = None
            self.model = None

        logger.info(f"✓ LLMInference ready: {self.model_id} (provider: {self.provider})")

    def _initialize_provider(self):
        """Initialize LLM provider instance"""
        if self.provider == 'ollama':
            try:
                model_mapping = {
                    # Qwen3 models (wit CoT)
                    "Qwen/Qwen3-4B-Instruct": "qwen3:4b",
                    "Qwen/Qwen3-8B-Instruct": "qwen3:8b",
                    "Qwen/Qwen3-14B-Instruct": "qwen3:14b",
                    "Qwen/Qwen3-32B-Instruct": "qwen3:32b",
                    "Qwen/Qwen3-30B-A3B-Instruct": "qwen3:30b-a3b",

                    # Qwen2.5 models (older models for tests - without CoT)
                    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5:3b",
                    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5:7b",
                    "Qwen/Qwen2.5-14B-Instruct": "qwen2.5:14b",
                }
                ollama_model = model_mapping.get(self.model_id)
                if ollama_model is None:
                    logger.warning(f"Model {self.model_id} not found in Ollama models. Using default model.")
                    ollama_model = "qwen3:4b"

                logger.info(f"🦙 Initializing Ollama with model: {ollama_model}")

                return OllamaProvider(
                    model_name=ollama_model,
                    host=getattr(settings.llm, 'ollama_host', 'http://localhost:11434'),
                )
            except ImportError as e:
                logger.error(f"❌ Ollama provider not found: {e}")
                logger.error("Install with: pip install ollama")
                logger.info("⚠️  Falling back to HuggingFace Transformers")
                self.provider = 'huggingface'
                return LLMModel()

        # mac / linux based systems, wont work on windows
        elif self.provider == 'vllm':
            from src.ragx.generation.providers.vllm_provider import VLLMProvider
            try:
                logger.info(f"⚡ Initializing vLLM with model: {self.model_id}")
                quantization = None
                if "Qwen" in self.model_id:
                    quantization = "awq"
                    logger.info("Using AWQ quantization for Qwen model")

                return VLLMProvider(
                    model_id=self.model_id,
                    tensor_parallel_size=settings.llm.tensor_parallel_size,
                    gpu_memory_utilization=settings.llm.gpu_memory_utilization,
                    max_model_len=settings.llm.max_model_len,
                    quantization=quantization,
                    trust_remote_code=True,
                )
            except ImportError as e:
                logger.error(f"❌ vLLM provider not found: {e}")
                logger.error("Install with: pip install vllm")
                logger.info("⚠️  Falling back to HuggingFace Transformers")
                self.provider = 'huggingface'
                return LLMModel()
        else:
            logger.info(f"🤗 Initializing HuggingFace Transformers with model: {self.model_id}")
            return LLMModel()

    def generate(
            self,
            prompt: str,
            temperature: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from prompt with optional streaming.

        Args:
            prompt: Input prompt string
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
        """
        temperature = temperature if temperature is not None else self.temperature
        max_new_tokens = max_new_tokens or self.max_new_tokens

        # Use new provider interface
        if self.provider in ['ollama', 'vllm'] and self._provider_instance:
            return self._provider_instance.generate(
                prompt=prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

        # for HuggingFace
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.llm_model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=settings.llm.top_p,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # remove prompt ,only return generated part
        prompt_length = len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        answer = generated_text[prompt_length:].strip()

        return answer

    def generate_stream(
            self,
            prompt: str,
            temperature: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Generate text from prompt with streaming.

        Args:
            prompt: Input prompt string
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate

        Yields:
            Generated text chunks as they are produced
        """
        temperature = temperature if temperature is not None else self.temperature
        max_new_tokens = max_new_tokens or self.max_new_tokens

        # Check if provider supports streaming
        if self.provider == 'ollama' and self._provider_instance:
            yield from self._provider_instance.generate_stream(
                prompt=prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            return

        # for HuggingFace
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.llm_model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=20.0
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=settings.llm.top_p,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()
