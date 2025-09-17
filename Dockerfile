FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements/pyproject
COPY pyproject.toml README.md ./

# Install uv for faster package management (optional)
RUN pip install --no-cache-dir uv

# Install Python dependencies
RUN uv pip install --system --no-cache \
    wikiextractor \
    qdrant-client \
    sentence-transformers \
    transformers \
    torch --index-url https://download.pytorch.org/whl/cpu \
    fastapi \
    uvicorn[standard] \
    pydantic \
    pydantic-settings \
    nltk \
    tqdm \
    jinja2 \
    numpy \
    requests \
    llama-index \
    llama-index-embeddings-huggingface \
    langchain \
    langchain-community \
    click \
    python-dotenv

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Copy source code
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/index /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command
CMD ["python", "-c", "print('RAGx container ready')"]