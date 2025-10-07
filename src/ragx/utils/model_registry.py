"""Singleton registry for shared model instances to avoid reloading."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton registry for caching model instances to avoid multiple loads."""

    _instance: Optional[ModelRegistry] = None

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models: Dict[str, Any] = {}
            logger.info("ModelRegistry singleton initialized")
        return cls._instance

    def get_or_create(
        self, 
        key: str, 
        factory_fn: Callable[[], Any],
        force_reload: bool = False,
    ) -> Any:
        """
        Get cached model or create new one using factory function.

        Args:
            key: Unique identifier for the model (e.g., model_id + device)
            factory_fn: Function to create the model if not cached
            force_reload: If True, reload the model even if cached

        Returns:
            Cached or newly created model instance
        """
        if force_reload and key in self._models:
            logger.info(f"Force reloading model: {key}")
            del self._models[key]

        if key not in self._models:
            logger.info(f"Loading model into registry: {key}")
            self._models[key] = factory_fn()
            logger.info(f"Model loaded and cached: {key}")
        else:
            logger.debug(f"Reusing cached model: {key}")

        return self._models[key]

    def get(self, key: str) -> Optional[Any]:
        """Get cached model without creating it."""
        return self._models.get(key)

    def has(self, key: str) -> bool:
        """Check if model is cached."""
        return key in self._models

    def remove(self, key: str) -> bool:
        """Remove model from cache."""
        if key in self._models:
            logger.info(f"Removing model from registry: {key}")
            del self._models[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached models."""
        count = len(self._models)
        self._models.clear()
        logger.info(f"ModelRegistry cleared ({count} models removed)")

    def list_cached(self) -> list[str]:
        """List all cached model keys."""
        return list(self._models.keys())


model_registry = ModelRegistry()