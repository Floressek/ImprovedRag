.PHONY: help install dev setup-qdrant download-wiki ingest search test-pipeline clean docker-build docker-up docker-down

# Variables
PY = python
PIP = pip
DOCKER_COMPOSE = docker-compose

# Default target
help:
	@echo "RAGx Makefile Commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make dev           - Install dev dependencies"
	@echo "  make setup-qdrant  - Start Qdrant container"
	@echo "  make download-wiki - Download Wikipedia dump"
	@echo "  make ingest        - Ingest Wikipedia into Qdrant"
	@echo "  make search        - Test search functionality"
	@echo "  make test-pipeline - Run end-to-end test"
	@echo "  make clean         - Clean cache and temp files"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start all services"
	@echo "  make docker-down   - Stop all services"

# Installation
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt || true
	$(PIP) install \
		wikiextractor \
		qdrant-client \
		sentence-transformers \
		transformers \
		torch \
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
	$(PY) -c "import nltk; nltk.download('punkt')"

dev:
	$(PIP) install --upgrade pip
	$(PIP) install pytest black isort ruff mypy ipython jupyter

# Qdrant setup
setup-qdrant:
	@echo "Starting Qdrant..."
	$(DOCKER_COMPOSE) up -d qdrant
	@echo "Waiting for Qdrant to be ready..."
	@sleep 5
	@curl -s http://localhost:6333/health | grep -q true && echo "✓ Qdrant is running" || echo "✗ Qdrant failed to start"

stop-qdrant:
	$(DOCKER_COMPOSE) stop qdrant

# Wikipedia ingestion pipeline
download-wiki:
	@echo "Downloading Wikipedia dump (small chunk for testing)..."
	$(PY) scripts/ingest_wiki.py \
		--download \
		--language en \
		--chunk-number 1 \
		--max-articles 1000

ingest: setup-qdrant
	@echo "Ingesting Wikipedia into Qdrant..."
	$(PY) scripts/ingest_wiki.py \
		--max-articles 1000 \
		--chunk-size 512 \
		--chunk-overlap 96 \
		--chunking-strategy semantic \
		--embedding-model thenlper/gte-multilingual-base \
		--use-prefixes \
		--recreate-collection

ingest-full: setup-qdrant
	@echo "Ingesting full Wikipedia chunk..."
	$(PY) scripts/ingest_wiki.py \
		--download \
		--language en \
		--chunk-number 1 \
		--max-articles 10000 \
		--chunk-size 512 \
		--chunk-overlap 96 \
		--chunking-strategy semantic \
		--embedding-model thenlper/gte-multilingual-base \
		--use-prefixes \
		--recreate-collection

# Search testing
search:
	@echo "Testing search..."
	$(PY) scripts/test_search.py "artificial intelligence" \
		--top-k 20 \
		--use-reranker \
		--rerank-top-k 5

search-simple:
	@echo "Testing search without reranker..."
	$(PY) scripts/test_search.py "machine learning" \
		--top-k 5

# End-to-end test
test-pipeline: setup-qdrant
	@echo "Running end-to-end test..."
	@echo "1. Downloading small Wikipedia sample..."
	$(PY) scripts/ingest_wiki.py --download --language en --chunk-number 1 --max-articles 100
	@echo "2. Ingesting data..."
	$(PY) scripts/ingest_wiki.py --max-articles 100 --recreate-collection
	@echo "3. Testing search..."
	$(PY) scripts/test_search.py "computer science" --top-k 5
	@echo "✓ Pipeline test complete!"

# Docker commands
docker-build:
	$(DOCKER_COMPOSE) build

docker-up:
	$(DOCKER_COMPOSE) up -d
	@echo "Waiting for services..."
	@sleep 10
	@echo "Services status:"
	@$(DOCKER_COMPOSE) ps

docker-down:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs -f

# API server
api:
	uvicorn src.ragx.api.main:app --host 0.0.0.0 --port 8000 --reload

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-data:
	rm -rf data/processed/wiki_extracted
	rm -f data/raw/*.bz2

clean-all: clean clean-data
	$(DOCKER_COMPOSE) down -v
	rm -rf models/
	rm -rf .cache/

# Development helpers
fmt:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

lint:
	ruff check src/ scripts/
	mypy src/ --ignore-missing-imports

# Quick start
quickstart: setup-qdrant download-wiki ingest
	@echo "✓ RAGx is ready! Try: make search"

# Status check
status:
	@echo "Checking system status..."
	@echo -n "Qdrant: "
	@curl -s http://localhost:6333/health 2>/dev/null | grep -q true && echo "✓ Running" || echo "✗ Not running"
	@echo -n "Python: "
	@$(PY) --version
	@echo -n "WikiExtractor: "
	@$(PY) -c "import wikiextractor; print('✓ Installed')" 2>/dev/null || echo "✗ Not installed"
	@echo -n "Qdrant Client: "
	@$(PY) -c "import qdrant_client; print('✓ Installed')" 2>/dev/null || echo "✗ Not installed"
	@echo -n "Sentence Transformers: "
	@$(PY) -c "import sentence_transformers; print('✓ Installed')" 2>/dev/null || echo "✗ Not installed"