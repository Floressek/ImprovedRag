.PHONY: help install dev setup-qdrant download-wiki extract-wiki ingest search status clean

PY = python
PIP = pip
DOCKER_COMPOSE = docker-compose
PIPELINE = $(PY) -m src.ragx.ingestion.pipelines.pipeline


help:
	@echo "RAGx Makefile Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install          - Install all dependencies"
	@echo "  make dev              - Install dev dependencies"
	@echo "  make setup-qdrant     - Start Qdrant container"
	@echo ""
	@echo "Wikipedia Pipeline:"
	@echo "  make download-wiki    - Download Polish Wikipedia dump"
	@echo "  make extract-wiki     - Extract Wikipedia dump using Docker"
	@echo "  make ingest-test      - Test ingestion (1k articles)"
	@echo "  make ingest-full      - Full ingestion (10k articles)"
	@echo "  make ingest-custom    - Custom ingestion (set MAX_ARTICLES=N)"
	@echo ""
	@echo "Search & Status:"
	@echo "  make search QUERY='...' - Search (example: make search QUERY='python')"
	@echo "  make status            - Check Qdrant status"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            - Clean cache files"
	@echo "  make clean-data       - Clean data files"
	@echo "  make clean-all        - Clean everything"
	@echo "  make fmt              - Format code (black + isort)"
	@echo "  make lint             - Lint code (ruff + mypy)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up        - Start all services"
	@echo "  make docker-down      - Stop all services"
	@echo "  make docker-logs      - View logs"

# ============================================================================
# Installation
# ============================================================================

install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt || true
	$(PIP) install \
		qdrant-client \
		sentence-transformers \
		transformers \
		torch \
		fastapi \
		uvicorn[standard] \
		pydantic \
		pydantic-settings \
		click \
		python-dotenv \
		llama-index \
		llama-index-embeddings-huggingface \
		langchain \
		langchain-community \
		tqdm \
		beautifulsoup4
	@echo "Installation complete!"

dev:
	@echo "Installing dev dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install pytest black isort ruff mypy ipython jupyter
	@echo "Dev dependencies installed!"

# ============================================================================
# Qdrant Setup
# ============================================================================

setup-qdrant:
	@echo "Starting Qdrant..."
	$(DOCKER_COMPOSE) up -d qdrant
	@echo "Waiting for Qdrant to be ready..."
	@powershell -Command "Start-Sleep -Seconds 3"
	@powershell -Command "try { $$response = Invoke-RestMethod -Uri 'http://localhost:6333/' -TimeoutSec 10; Write-Host 'Qdrant is running' } catch { Write-Host 'Qdrant failed to start' }"

stop-qdrant:
	@echo "Stopping Qdrant..."
	$(DOCKER_COMPOSE) stop qdrant

# ============================================================================
# Wikipedia Pipeline
# ============================================================================

download-wiki:
	@echo "Downloading Polish Wikipedia dump..."
	@if not exist "data\raw" mkdir data\raw
	@powershell -Command "$$confirm = Read-Host 'This will download ~2-3 GB. Continue? (y/N)'; if ($$confirm -ne 'y') { Write-Host 'Download cancelled'; exit 1 }"
	@curl -L "https://dumps.wikimedia.org/plwiki/latest/plwiki-latest-pages-articles-multistream.xml.bz2" -o data/raw/plwiki-latest.xml.bz2
	@echo "Wikipedia dump downloaded!"

extract-wiki:
	@echo "Extracting Wikipedia dump using Docker..."
	@powershell -Command "docker run --rm -v \"$${PWD}/data:/data\".Replace('\', '/') python:3.11-slim bash -c \"apt-get update -qq && apt-get install -y -qq git && pip install -q git+https://github.com/attardi/wikiextractor.git && mkdir -p /data/processed/wiki_extracted && wikiextractor /data/raw/plwiki-latest.xml.bz2 --output /data/processed/wiki_extracted --bytes 1M --processes 8 --json --no-templates\""
	@echo "Wikipedia extraction complete!"

# ============================================================================
# Ingestion (Simplified - uses .env for most settings)
# ============================================================================

ingest-test: setup-qdrant
	@echo "Test ingestion (1k articles)..."
	$(PIPELINE) ingest data/processed/wiki_extracted \
		--max-articles 1000 \
		--recreate-collection \
		--batch-size 250
	@echo "Test ingestion complete!"

ingest-full: setup-qdrant
	@echo "Full ingestion (10k articles)..."
	$(PIPELINE) ingest data/processed/wiki_extracted \
		--max-articles 1000000 \
		--recreate-collection \
		--batch-size 400
	@echo "Full ingestion complete!"

ingest-custom: setup-qdrant
	@echo "Custom ingestion ($(MAX_ARTICLES) articles)..."
	$(PIPELINE) ingest data/processed/wiki_extracted \
		--max-articles $(MAX_ARTICLES) \
		--recreate-collection \
		--batch-size 100
	@echo "Custom ingestion complete!"

# Override chunk settings for experiments
ingest-experiment: setup-qdrant
	@echo "Experimental ingestion with custom settings..."
	$(PIPELINE) ingest data/processed/wiki_extracted \
		--max-articles $(MAX_ARTICLES) \
		--chunk-size $(CHUNK_SIZE) \
		--chunk-overlap $(CHUNK_OVERLAP) \
		--recreate-collection
	@echo "Experimental ingestion complete!"

# Resume ingestion from last processed file
ingest-resume: setup-qdrant
	@echo "Resuming ingestion from last processed file..."
	python -m src.ragx.ingestion.pipelines.pipeline ingest data/processed/wiki_extracted --resume --max-articles 10000

# Start ingestion from specific file (use: make ingest-from FILE=wiki_00)
ingest-from: setup-qdrant
	@echo "Starting ingestion from file: $(FILE)"
	python -m src.ragx.ingestion.pipelines.pipeline ingest data/processed/wiki_extracted --start-from-file $(FILE) --max-articles 10000

# ============================================================================
# Search & Status
# ============================================================================

search:
	@echo "Searching: '$(QUERY)'..."
	$(PIPELINE) search "$(QUERY)" --top-k 15
	@echo ""

search-more:
	@echo "Searching: '$(QUERY)' (top 10)..."
	$(PIPELINE) search "$(QUERY)" --top-k 20
	@echo ""

status:
	@echo "Checking system status..."
	@echo ""
	$(PIPELINE) status
	@echo ""
	@echo "System Check:"
	@echo "Python:   $$($(PY) --version)"
	@powershell -Command "try { $$null = Invoke-RestMethod -Uri 'http://localhost:6333/' -TimeoutSec 2; Write-Host 'Qdrant:   Running' } catch { Write-Host 'Qdrant:   Not running' }"

# Show detailed ingestion status with file history
status-detailed:
	@echo "Detailed ingestion status..."
	python -m src.ragx.ingestion.pipelines.pipeline status --show-files

# Clear ingestion progress (start fresh)
clear-progress:
	@echo "Clearing ingestion progress..."
	@rm -f data/.ingestion_progress.json
	@echo "Progress cleared. Next ingestion will start from scratch."

# ============================================================================
# Docker Commands
# ============================================================================

docker-build:
	@echo "Building Docker images..."
	$(DOCKER_COMPOSE) build

docker-up:
	@echo "Starting all services..."
	$(DOCKER_COMPOSE) up -d
	@echo "Waiting for services..."
	@powershell -Command "Start-Sleep -Seconds 5"
	@$(DOCKER_COMPOSE) ps

docker-down:
	@echo "Stopping all services..."
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs -f

# ============================================================================
# Cleaning
# ============================================================================

clean:
	@echo "Cleaning cache files..."
	@if exist __pycache__ rmdir /s /q __pycache__
	@for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@for /d /r %%d in (.pytest_cache) do @if exist "%%d" rmdir /s /q "%%d"
	@for /d /r %%d in (.mypy_cache) do @if exist "%%d" rmdir /s /q "%%d"
	@for /d /r %%d in (.ruff_cache) do @if exist "%%d" rmdir /s /q "%%d"
	@del /s /q *.pyc 2>nul
	@echo "Cache cleaned!"

clean-data:
	@echo "Cleaning data files..."
	@if exist data\processed\wiki_extracted rmdir /s /q data\processed\wiki_extracted
	@if exist data\raw\*.bz2 del /q data\raw\*.bz2
	@echo "Data cleaned!"

clean-all: clean clean-data
	@echo "Deep cleaning..."
	$(DOCKER_COMPOSE) down -v
	@if exist models rmdir /s /q models
	@if exist .cache rmdir /s /q .cache
	@echo "Everything cleaned!"

# ============================================================================
# Development Helpers
# ============================================================================

fmt:
	@echo "Formatting code..."
	black src/ scripts/ tests/
	isort src/ scripts/ tests/
	@echo "Code formatted!"

lint:
	@echo "Linting code..."
	ruff check src/ scripts/
	mypy src/ --ignore-missing-imports
	@echo "Linting complete!"

# ============================================================================
# Quick Start Workflows
# ============================================================================

quickstart: setup-qdrant download-wiki extract-wiki ingest-test
	@echo ""
	@echo "RAGx is ready!"
	@echo ""
	@echo "Try searching:"
	@echo "  make search QUERY='python'"
	@echo "  make search QUERY='sztuczna inteligencja'"
	@echo ""

full-pipeline: setup-qdrant download-wiki extract-wiki ingest-full
	@echo ""
	@echo "Full pipeline complete!"
	@echo ""
	@echo "Collection info:"
	@$(PIPELINE) status
	@echo ""

# ============================================================================
# Examples
# ============================================================================

examples:
	@echo "Usage Examples:"
	@echo ""
	@echo "1. Quick Start (1k articles):"
	@echo "   make quickstart"
	@echo ""
	@echo "2. Full Pipeline (10k articles):"
	@echo "   make full-pipeline"
	@echo ""
	@echo "3. Custom Ingestion:"
	@echo "   make ingest-custom MAX_ARTICLES=5000"
	@echo ""
	@echo "4. Experiment with Settings:"
	@echo "   make ingest-experiment MAX_ARTICLES=1000 CHUNK_SIZE=1024 CHUNK_OVERLAP=128"
	@echo ""
	@echo "5. Search:"
	@echo "   make search QUERY='machine learning'"
	@echo "   make search-more QUERY='python'"
	@echo ""
	@echo "6. Check Status:"
	@echo "   make status"
	@echo ""