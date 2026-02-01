FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir uv

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

RUN python -c "import nltk; nltk.download('punkt')"

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN mkdir -p /app/data/raw /app/data/processed /app/data/index /app/logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH

CMD ["python", "-c", "print('RAGx container ready')"]