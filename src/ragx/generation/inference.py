from __future__ import annotations

import logging
from typing import Optional, Iterator

import torch
from transformers import TextIteratorStreamer
from threading import Thread

from src.ragx.generation.model import LLMModel
from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


class LLMInference:
    """LLM inference with streaming support."""

    def __init__(
            self,
            llm_model: Optional[LLMModel] = None,
            temperature: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
    ):
        self.llm_model = llm_model or LLMModel()
        self.temperature = temperature if temperature is not None else settings.llm.temperature
        self.max_new_tokens = max_new_tokens or settings.llm.max_new_tokens

        self.tokenizer = self.llm_model.get_tokenizer()
        self.model = self.llm_model.get_model()

        logger.info(f"LLMInference initialized with model {self.llm_model.model_id}")

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
