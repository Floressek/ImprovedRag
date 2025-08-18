"""FastAPI app placeholder with lifespan management comments.
No external dependencies are imported to keep the scaffold light.
"""
from __future__ import annotations
from typing import Callable, Any, List, Tuple


class DummyApp:
    """Minimal stand-in for a web framework app.

    Stores registered routes as tuples of (path, handler_name).
    """

    def __init__(self) -> None:
        self.routes: List[Tuple[str, str]] = []
        self.middlewares: List[str] = []

    def add_route(self, path: str, handler: Callable[..., Any]) -> None:
        self.routes.append((path, getattr(handler, "__name__", "handler")))

    def add_middleware(self, name: str) -> None:
        self.middlewares.append(name)


app = DummyApp()


def on_startup() -> None:
    """Placeholder for startup events."""
    # e.g., connect to vector store, load models
    pass


def on_shutdown() -> None:
    """Placeholder for shutdown events."""
    # e.g., close connections, flush logs
    pass


if __name__ == "__main__":
    on_startup()
    print("API placeholder app initialized with", len(app.routes), "routes")
    on_shutdown()
