from __future__ import annotations
import logging
from typing import Optional, Iterator
import requests

from src.ragx.utils.settings import settings

logger = logging.getLogger(__name__)


class APIProvider:
    """OpenAI-compatible API provider (LM Studio, OpenAI, etc.)"""

    def __init__(
            self,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            model_name: Optional[str] = None,
            timeout: int = 120,
    ):
        self.base_url = base_url.rstrip('/') or settings.llm.base_url.rstrip('/')
        self.api_key = api_key or settings.llm.api_key
        self.model_name = model_name or settings.llm.api_model_name
        self.timeout = timeout

        logger.info(f"APIProvider: {base_url}, model: {model_name}")


    def generate(
            self,
            prompt: str,
            temperature: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
            chain_of_thought_enabled: Optional[bool] = None,
    ) -> str:
        """
        Generate text from the API provider.

        Args:
            prompt (str): The input prompt to generate text from.
            temperature (Optional[float]): Sampling temperature.
            max_new_tokens (Optional[int]): Maximum number of new tokens to generate.
            chain_of_thought_enabled (Optional[float]): Unused parameter for compatibility.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature or settings.llm.temperature,
            "max_tokens": max_new_tokens or settings.llm.max_new_tokens,
            "enable_thinking": False,
        }

        # if chain_of_thought_enabled is not None:
        #     payload["extra_body"] = {"enable_thinking": chain_of_thought_enabled}

        url = f"{self.base_url}/chat/completions"
        logger.info(f"🔍 Making request to: {url}")
        logger.info(f"🔍 Model: {self.model_name}")
        if chain_of_thought_enabled is not None:
            logger.info(f"🔍 enable_thinking: {chain_of_thought_enabled}")

        response = None
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            logger.info(f"📋 Full API response structure: {list(data.keys())}")
            if 'choices' in data and len(data['choices']) > 0:
                logger.info(f"📋 Message keys: {list(data['choices'][0].get('message', {}).keys())}")

            generated_text = data['choices'][0]['message']['content']

            # Extract thinking/reasoning if present
            if chain_of_thought_enabled:
                thinking = None
                message = data['choices'][0]['message']
                if 'thinking' in message:
                    thinking = message['thinking']
                elif 'reasoning' in message:
                    thinking = message['reasoning']
                elif 'thoughts' in message:
                    thinking = message['thoughts']

                if thinking:
                    logger.info(f"Thinking process: {thinking}")
                else:
                    logger.warning("Chain of thought enabled but no thinking data found in response")

            return generated_text

        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ APIProvider HTTP error: {e}")
            logger.error(f"❌ Response status: {response.status_code}")
            logger.error(f"❌ Response body: {response.text}")
            return ""
        except Exception as e:
            logger.error(f"❌ APIProvider generate error: {e}")
        return ""

    def generate_stream(
            self,
            prompt: str,
            temperature: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """
        Generate text from the API provider - streaming.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature or settings.llm.temperature,
            "max_tokens": max_new_tokens or settings.llm.max_new_tokens,
            "stream": True,
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        if line.strip() == 'data: [DONE]':
                            break
                        try:
                            import json
                            data = json.loads(line[6:])
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except:
                            continue

        except Exception as e:
            logger.error(f"Stream failed: {e}")
