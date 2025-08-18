"""Dependency injection placeholders for API."""
from __future__ import annotations


def get_settings() -> dict:
    """Return minimal settings dict placeholder."""
    return {"env": "development", "log_level": "INFO"}


def get_index() -> object:
    """Return a placeholder index object."""
    class _Index:
        def search(self, *_args, **_kwargs):
            return []
    return _Index()
