from __future__ import annotations

from typing import Dict, Any
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models: Dict[str, bool]
    collection: Dict[str, Any]