"""Model loading and management (placeholder)."""
from .providers.factory import get_provider


def load_model(kind: str = "hf"):
    return get_provider(kind)