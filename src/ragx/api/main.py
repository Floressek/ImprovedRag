from __future__ import annotations

import logging
import time
import warnings
from contextlib import asynccontextmanager
from fastapi import Request

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.ragx.api.routers import chat, search, health
from src.ragx.api.dependencies import (
    get_baseline_pipeline,
    get_enhanced_pipeline,
)
from src.ragx.utils.logging_config import setup_logging
from src.ragx.utils.settings import settings

setup_logging(level=settings.app.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    logger.info("🚀 Starting RAGx API server...")

    # Warmup models
    logger.info("Warming up models...")
    baseline = get_baseline_pipeline()
    enhanced = get_enhanced_pipeline()

    logger.info("✓ Models ready")
    logger.info(f"✓ Collection: {settings.qdrant.collection_name}")
    logger.info(f"✓ Embedder: {settings.embedder.model_id}")
    logger.info(f"✓ Reranker: {settings.reranker.model_id}")
    logger.info(f"✓ LLM: {settings.llm.model_id}")

    yield

    logger.info("Shutting down RAGx API server...")


app = FastAPI(
    title="RAGx API",
    description="RAGx API service",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    logger.info(f"📨 Incoming: {request.method} {request.url.path}")

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"📤 Response: {request.method} {request.url.path} "
        f"Status={response.status_code} Time={process_time:.2f}ms"
    )

    return response


# Routers
app.include_router(chat.router)

app.include_router(search.router)
app.include_router(health.router)


@app.get("/api")
async def root():
    """Root endpoint."""
    return {
        "name": "RAGx API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "baseline": "/ask/baseline",
            # "baseline_stream": "/ask/baseline/stream",
            # "enhanced": "/ask/enhanced",
            # "enhanced_stream": "/ask/enhanced/stream",
            # "search": "/search",
            # "rerank": "/rerank",
            # "health": "/health",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.ragx.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload_excludes=["*.pyc", "__pycache__"],
        log_level=settings.app.log_level.lower(),
        access_log=True
    )
