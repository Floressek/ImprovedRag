import logging
import json
from typing import Dict, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.ragx.evaluation.models import PipelineConfig

logger = logging.getLogger(__name__)


class RAGAPIClient:
    """Client for calling RAG API with retry logic."""

    def __init__(
            self,
            api_base_url: str,
            timeout: int = 120,
            retry_total: int = 3,
            retry_backoff: int = 2,
    ):
        """
        Initialize API client.

        Args:
            api_base_url: Base URL for RAG API (e.g., http://localhost:8000)
            timeout: API request timeout in seconds (default: 120)
            retry_total: Max number of retries for failed requests (default: 3)
            retry_backoff: Backoff factor for retries in seconds (default: 2)
        """
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout

        # Configure retry strategy for network resilience
        retry_strategy = Retry(
            total=retry_total,
            backoff_factor=retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP codes
            allowed_methods=["POST"],  # Only retry POST requests
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info(
            f"Initialized API client for {self.api_base_url} "
            f"(timeout: {timeout}s, retry: {retry_total}x with backoff {retry_backoff}s)"
        )

    def __del__(self):
        """Clean up HTTP session on destruction."""
        if hasattr(self, 'session'):
            try:
                self.session.close()
                logger.debug("Closed HTTP session")
            except Exception as e:
                logger.warning(f"Error closing session: {e}")

    def call_ablation_endpoint(
            self,
            query: str,
            config: PipelineConfig,
    ) -> Dict[str, Any]:
        """
        Call RAG API /eval/ablation endpoint with specific configuration.

        Args:
            query: User question
            config: Pipeline configuration

        Returns:
            API response dict with answer, sources, metadata

        Raises:
            requests.exceptions.RequestException: On API errors
        """
        url = f"{self.api_base_url}/eval/ablation"

        # Flatten config into request (no nested "config" object)
        payload = {
            "query": query,
            **config.to_dict(),  # Use config's to_dict() method
        }

        logger.debug(f"Sending request to {url}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        response = self.session.post(url, json=payload, timeout=self.timeout)

        # Log response details before raising
        if not response.ok:
            logger.error(f"API returned {response.status_code}: {response.text}")

        response.raise_for_status()

        return response.json()
