"""YAML loading utilities (placeholder)."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists() or yaml is None:
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
