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
	@echo "  make download-pl-wiki - Download Wikipedia dump"
	@echo "  make extract-wiki-docker  - Extract Wikipedia dump using Docker"
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
	@powershell -Command "Start-Sleep -Seconds 3"
	@powershell -Command "try { $$response = Invoke-RestMethod -Uri 'http://localhost:6333/' -TimeoutSec 10; Write-Host '✓ Qdrant is running' } catch { Write-Host '✗ Qdrant failed to start' }"

stop-qdrant:
	$(DOCKER_COMPOSE) stop qdrant

# Wikipedia ingestion pipeline
download-pl-wiki:
	@echo "Downloading Polish Wikipedia dump..."
	mkdir -p data/raw/pl_wiki_dump
	@echo "This will download a large file (several GB). Continue? [y/N]"
	@read -r REPLY; if [ "$$REPLY" = "y" ] || [ "$$REPLY" = "Y" ]; then \
		curl -L "https://dumps.wikimedia.org/plwiki/20250601/plwiki-20250601-pages-articles-multistream.xml.bz2" \
			-o data/raw/pl_wiki_dump/plwiki-20250601-pages-articles-multistream.xml.bz2; \
		echo "✓ Polish Wikipedia dump downloaded!"; \
	else \
		echo "Download cancelled."; \
	fi


extract-wiki-docker-fixed:
	@echo "Extracting Wikipedia dump using Docker (Windows fixed)..."
	@powershell -Command "docker run --rm -v \"$${PWD}/data:/data\".Replace('\', '/') python:3.11-slim bash -lc \"apt-get update && apt-get install -y git && pip install -q git+https://github.com/attardi/wikiextractor.git@ab8988ebfa9e4557411f3d4c0f4ccda139e18875 && mkdir -p /data/processed/wiki_extracted && wikiextractor /data/raw/pl_wiki_dump/plwiki-20250601-pages-articles-multistream.xml.bz2 --output /data/processed/wiki_extracted --bytes 1M --processes 8 --json --no-templates\""
	@echo "✓ Wikipedia extraction complete!"

ingest: setup-qdrant
	@echo "Ingesting Wikipedia into Qdrant..."
	$(PY) scripts/ingest_wiki.py \
		--max-articles 1000 \
		--chunk-size 512 \
		--chunk-overlap 96 \
		--chunking-strategy semantic \
		--embedding-model Alibaba-NLP/gte-multilingual-base \
		--use-prefixes \
		--recreate-collection

ingest-full:
	@echo "Ingesting full Wikipedia chunk..."
	$(PY) scripts/ingest_wiki.py \
		--source data\\processed\\wiki_extracted \
		--max-articles 10000 \
		--chunk-size 512 \
		--chunk-overlap 96 \
		--chunking-strategy semantic \
		--embedding-model Alibaba-NLP/gte-multilingual-base \
		--use-prefixes \
		--recreate-collection


# Search testing
#search:
#	@echo "Testing search..."
#	$(PY) scripts/test_search.py "artificial intelligence" \
#		--top-k 20 \
#		--use-reranker \
#		--rerank-top-k 5

search:
	@echo "Testing search..."
	python -m src.ragx.ingestion.pipeline search "$(QUERY)" \
		--top-k $(TOP_K) \
		--collection-name ragx_documents \
		--use-prefixes

search-simple:
	@echo "Testing search without reranker..."
	$(PY) scripts/test_search.py "machine learning" \
		--top-k 5

# End-to-end test
#test-pipeline: setup-qdrant
#	@echo "Running end-to-end test..."
#	@echo "1. Downloading small Wikipedia sample..."
#	$(PY) scripts/ingest_wiki.py --download --language en --chunk-number 1 --max-articles 100
#	@echo "2. Ingesting data..."
#	$(PY) scripts/ingest_wiki.py --max-articles 100 --recreate-collection
#	@echo "3. Testing search..."
#	$(PY) scripts/test_search.py "computer science" --top-k 5
#	@echo "✓ Pipeline test complete!"

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
#api:
#	uvicorn src.ragx.api.main:app --host 0.0.0.0 --port 8000 --reload

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
	@echo "Qdrant:" && (curl -s http://localhost:6333/ >nul 2>&1 && echo "  OK Running" || echo "  ERROR Not running")
	@echo "Python:" && $(PY) --version
	@echo "WikiExtractor:" && ($(PY) -c "import wikiextractor; print('  OK Installed')" 2>nul || echo "  ERROR Not installed")
	@echo "Qdrant Client:" && ($(PY) -c "import qdrant_client; print('  OK Installed')" 2>nul || echo "  ERROR Not installed")
	@echo "Sentence Transformers:" && ($(PY) -c "import sentence_transformers; print('  OK Installed')" 2>nul || echo "  ERROR Not installed")