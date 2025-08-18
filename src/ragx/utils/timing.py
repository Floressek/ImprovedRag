"""Timing utilities (placeholder)."""
import time
from functools import wraps
from typing import Callable, TypeVar, Any

F = TypeVar("F", bound=Callable[..., Any])

def timed(func: F) -> F:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            duration = (time.perf_counter() - start) * 1000
            print(f"{func.__name__} took {duration:.2f} ms")
    return wrapper  # type: ignore[misc]
