"""Text generation (placeholder)."""
from .model import load_model


def generate(text: str, provider: str = "hf") -> str:
    model = load_model(provider)
    return model.generate(text)
