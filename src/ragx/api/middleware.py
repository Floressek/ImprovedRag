"""Middleware placeholders: logging, timing, error handling."""
from __future__ import annotations


def request_logger_middleware() -> str:
    """Return a placeholder middleware name."""
    return "request_logger"


def timing_middleware() -> str:
    return "timing"


def error_handling_middleware() -> str:
    return "error_handler"
