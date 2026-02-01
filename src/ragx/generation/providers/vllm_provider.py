from __future__ import annotations

import logging
from typing import Optional

from vllm import LLM, SamplingParams

from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


class VLLMProvider:
    """LLM provider using vLLM. - inference optimized LLM serving library
    Optional provider, since api_provider is more flexible and also is capable of using locally hosted models on vllm.
    """

    def __init__(
            self,
            model_id: Optional[str] = None,
            tensor_parallel_size: Optional[int] = None,
            gpu_memory_utilization: Optional[float] = None,
            max_model_len: Optional[int] = None,
            quantization: Optional[str] = None,
            trust_remote_code: Optional[bool] = None,
    ):
        """
         Initialize vLLM engine.

         Args:
             model_id: HuggingFace model ID
             tensor_parallel_size: Number of GPUs for tensor parallelism
             gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
             max_model_len: Maximum context length
             quantization: Quantization method ('awq', 'gptq', or None)
             trust_remote_code: Trust remote code
         """
        self.model_id = model_id or settings.llm.model_id
        self.tensor_parallel_size = tensor_parallel_size or settings.llm.tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization or settings.llm.gpu_memory_utilization
        self.max_model_len = max_model_len or settings.llm.max_model_len
        self.quantization = quantization or settings.llm.quantization
        self.trust_remote_code = trust_remote_code or settings.llm.trust_remote_code

        logger.info(f"Loading vLLM model: {model_id}")
        logger.info(f"  - Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"  - GPU memory utilization: {gpu_memory_utilization}")
        logger.info(f"  - Max model length: {max_model_len}")

        self.llm = LLM(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            trust_remote_code=self.trust_remote_code,
            enforce_eager=False,  # use CUDA for speedup
            disable_log_stats=False  # monitoring for now enabled
        )

        logger.info(f"vLLM model loaded: {model_id}")

    def generate(
            self,
            prompt: str,
            temperature: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
            repetition_penalty: Optional[float] = None,
            top_p: Optional[float] = None,
    ) -> str:
        """
        Generate text from prompt. STREAMING IS NOT SUPPORTED.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            repetition_penalty: Repetition penalty

        Returns:
            Generated text
        """
        sampling_params = SamplingParams(
            temperature=temperature or settings.llm.temperature,
            top_p=top_p or settings.llm.top_p,
            repetition_penalty=repetition_penalty or settings.llm.repetition_penalty,
            max_tokens=max_new_tokens or settings.llm.max_new_tokens,
        )

        outputs = self.llm.generate([prompt], sampling_params)

        # returns list of outputs
        generated_text = outputs[0].outputs[0].text

        return generated_text

    def batch_generate(
            self,
            prompts: list[str],
            temperature: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
    ) -> list[str]:
        """
        Generate for multiple prompts in batch (vLLM's strength!).

        Args:
            prompts: List of prompts
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of generated texts
        """
        sampling_params = SamplingParams(
            temperature=temperature or settings.llm.temperature,
            max_tokens=max_new_tokens or settings.llm.max_new_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        return [outputs.outputs[0].text for outputs in outputs]
