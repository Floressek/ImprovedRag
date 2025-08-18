"""Pydantic-like settings placeholder that can load from .env and YAML in future.
Currently provides simple environment-backed defaults.
"""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    data_dir: str = os.getenv("DATA_DIR", "./data")
    raw_dir: str = os.getenv("RAW_DATA_DIR", "./data/raw")
    processed_dir: str = os.getenv("PROCESSED_DATA_DIR", "./data/processed")
    index_dir: str = os.getenv("INDEX_DIR", "./data/index")


def load_settings() -> Settings:
    return Settings()
