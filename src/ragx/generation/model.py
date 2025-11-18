from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.ragx.utils.settings import settings
from src.ragx.utils.model_registry import model_registry

logger = logging.getLogger(__name__)


class LLMModel:
    """LLM model wrapper for text generation."""

    def __init__(
            self,
            model_id: Optional[str] = None,
            device: Optional[str] = None,
            load_in_4bit: Optional[bool] = None,
            trust_remote_code: bool = True,
            **kwargs,
    ):
        """
        Initialize LLMModel with optional overrides.
        If any parameter is None, it will be loaded from settings.

        Args:
            model_id: Model ID or path (default: from settings)
            device: Device to run the model on ('cpu', 'cuda', 'auto') (default: from settings)
            load_in_4bit: Whether to load the model in 8-bit precision (default: from settings)
            trust_remote_code: Whether to trust remote code in model
            kwargs: Additional keyword arguments for model loading
        """
        self.model_id = model_id or settings.llm.model_id
        self.load_in_4bit = load_in_4bit if load_in_4bit is not None else settings.llm.load_in_4bit

        logger.warning(f"🔍 DEBUG: load_in_4bit parameter = {load_in_4bit}")
        logger.warning(f"🔍 DEBUG: settings.llm.load_in_4bit = {settings.llm.load_in_4bit}")
        logger.warning(f"🔍 DEBUG: self.load_in_4bit = {self.load_in_4bit}")
        device = device or settings.llm.device
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading LLM model: {self.model_id} on {self.device}")

        cache_key = f"llm:{self.model_id}:{self.device}:{self.load_in_4bit}"

        def _create_tokenizer() -> AutoTokenizer:
            return AutoTokenizer.from_pretrained(
                self.model_id,
                use_fast=True,
                trust_remote_code=trust_remote_code,
                cache_dir=settings.huggingface.hf_hub_cache,
            )

        def _create_model():
            if self.load_in_4bit and self.device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_type=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                return AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=trust_remote_code,
                    cache_dir=settings.huggingface.hf_hub_cache,
                    **kwargs,
                )
            else:
                return AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=trust_remote_code,
                    cache_dir=settings.huggingface.hf_hub_cache,
                    **kwargs,
                )

        self.tokenizer = model_registry.get_or_create(
            f"tokenizer:{self.model_id}",
            _create_tokenizer,
        )
        self.model = model_registry.get_or_create(
            cache_key,
            _create_model,
        )

        # Ensure tokenizer has a pad token, if not, set it to eos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"LLM loaded: {self.model_id} (4bit={self.load_in_4bit})")

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model
