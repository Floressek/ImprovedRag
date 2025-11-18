.PHONY: help install dev setup-qdrant download-wiki extract-wiki ingest search status clean

PY = python
PIP = pip
DOCKER_COMPOSE = docker-compose
PIPELINE = $(PY) -m src.ragx.ingestion.pipelines.pipeline

# Evaluation settings (can override: make eval-run NUM_QUESTIONS=50)
NUM_QUESTIONS ?= 20
RUN_ID ?= study_$(shell powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'")
CHECKPOINT_DIR ?= checkpoints
EVAL_OUTPUT ?= results/ablation_$(shell powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'").json


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
	@echo "RAGAS Evaluation (NEW!):"
	@echo "  make eval-help        - Show detailed evaluation help"
	@echo "  make eval-generate    - Generate test questions (default: 100)"
	@echo "  make eval-api         - Start RAG API server"
	@echo "  make eval-run         - Run ablation study with checkpointing"
	@echo "  make eval-resume      - Resume from checkpoint (set RUN_ID=...)"
	@echo "  make eval-quick       - Quick test (10 questions, 3 configs)"
	@echo "  make eval-clean       - Clean checkpoints and results"
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

# ============================================================================
# RAGAS Evaluation & Ablation Study
# ============================================================================

eval-help:
	@echo "RAGAS Evaluation Commands - Detailed Help"
	@echo ""
	@echo "============================================================================"
	@echo "1. GENERATE TEST QUESTIONS"
	@echo "============================================================================"
	@echo "Generate questions from Wikipedia for RAGAS evaluation."
	@echo ""
	@echo "Usage:"
	@echo "  make eval-generate                  # Default: 100 questions"
	@echo "  make eval-generate NUM_QUESTIONS=50 # Custom: 50 questions"
	@echo ""
	@echo "Output:"
	@echo "  data/eval/questions_100.jsonl"
	@echo ""
	@echo "============================================================================"
	@echo "2. START API SERVER"
	@echo "============================================================================"
	@echo "Start FastAPI server for ablation study to test against."
	@echo ""
	@echo "Usage:"
	@echo "  make eval-api    # Runs on http://localhost:8000"
	@echo ""
	@echo "Leave this running in a separate terminal!"
	@echo ""
	@echo "============================================================================"
	@echo "3. RUN ABLATION STUDY (WITH CHECKPOINT)"
	@echo "============================================================================"
	@echo "Run full ablation study with automatic checkpoint saving."
	@echo ""
	@echo "Usage:"
	@echo "  make eval-run                        # Default: 100 questions, all configs"
	@echo "  make eval-run NUM_QUESTIONS=50       # Custom: 50 questions"
	@echo "  make eval-run RUN_ID=my_test_001     # Custom run ID"
	@echo ""
	@echo "Features:"
	@echo "  - Auto-saves checkpoint after each config"
	@echo "  - Can resume with Ctrl+C"
	@echo "  - Progress bars with tqdm"
	@echo "  - Retry logic for failed questions"
	@echo ""
	@echo "Output:"
	@echo "  results/ablation_YYYYMMDD_HHMMSS.json"
	@echo "  checkpoints/checkpoint_RUNID.json"
	@echo ""
	@echo "============================================================================"
	@echo "4. RESUME FROM CHECKPOINT"
	@echo "============================================================================"
	@echo "Resume interrupted test from checkpoint."
	@echo ""
	@echo "Usage:"
	@echo "  make eval-resume RUN_ID=study_20250118_153045"
	@echo ""
	@echo "Required:"
	@echo "  - RUN_ID must match original test"
	@echo "  - Checkpoint file must exist in checkpoints/"
	@echo ""
	@echo "============================================================================"
	@echo "5. QUICK TEST"
	@echo "============================================================================"
	@echo "Fast test with minimal questions and configs (for debugging)."
	@echo ""
	@echo "Usage:"
	@echo "  make eval-quick    # 10 questions, 3 configs (~2-5 min)"
	@echo ""
	@echo "Configs tested:"
	@echo "  - baseline"
	@echo "  - full_no_cove"
	@echo "  - full_cove_auto"
	@echo ""
	@echo "============================================================================"
	@echo "6. CLEAN EVALUATION DATA"
	@echo "============================================================================"
	@echo "Remove checkpoints and results."
	@echo ""
	@echo "Usage:"
	@echo "  make eval-clean    # Removes checkpoints/ and results/"
	@echo ""
	@echo "============================================================================"
	@echo ""
	@echo "TYPICAL WORKFLOW:"
	@echo ""
	@echo "  # Terminal 1: Generate questions"
	@echo "  make eval-generate NUM_QUESTIONS=100"
	@echo ""
	@echo "  # Terminal 2: Start API"
	@echo "  make eval-api"
	@echo ""
	@echo "  # Terminal 1: Run tests"
	@echo "  make eval-run"
	@echo ""
	@echo "  # If interrupted, resume:"
	@echo "  make eval-resume RUN_ID=study_20250118_153045"
	@echo ""
	@echo "VARIABLES YOU CAN OVERRIDE:"
	@echo "  NUM_QUESTIONS     Number of questions to generate/test (default: 100)"
	@echo "  RUN_ID            Unique identifier for checkpoint (auto-generated)"
	@echo "  CHECKPOINT_DIR    Directory for checkpoints (default: checkpoints)"
	@echo "  EVAL_OUTPUT       Output path for results (auto-generated)"
	@echo ""

eval-generate:
	@echo "Generating $(NUM_QUESTIONS) test questions..."
	@if not exist "data\eval" mkdir data\eval
	$(PY) scripts/generate_questions.py \
		--num-questions $(NUM_QUESTIONS) \
		--data-dir data/processed/wiki_extracted \
		--output data/eval/questions_$(NUM_QUESTIONS).jsonl \
		--show-samples 3
	@echo ""
	@echo "Questions generated!"
	@echo "Output: data/eval/questions_$(NUM_QUESTIONS).jsonl"
	@echo ""

eval-api:
	@echo "Starting RAG API server..."
	@echo "API will be available at: http://localhost:8080"
	@echo "Docs: http://localhost:8080/docs"
	@echo ""
	@echo "Press Ctrl+C to stop"
	@echo ""
	$(PY) -m src.ragx.api.main

eval-run:
	@echo "Running ablation study..."
	@echo "Questions: $(NUM_QUESTIONS)"
	@echo "Run ID: $(RUN_ID)"
	@echo "Checkpoint dir: $(CHECKPOINT_DIR)"
	@echo ""
	@if not exist "$(CHECKPOINT_DIR)" mkdir $(CHECKPOINT_DIR)
	@if not exist "results" mkdir results
	$(PY) scripts/run_ablation_study.py \
		--questions data/eval/questions_$(NUM_QUESTIONS).jsonl \
		--output $(EVAL_OUTPUT) \
		--checkpoint-dir $(CHECKPOINT_DIR) \
		--run-id $(RUN_ID) \
		--api-url http://localhost:8080
	@echo ""
	@echo "Ablation study complete!"
	@echo "Results: $(EVAL_OUTPUT)"
	@echo "Checkpoint: $(CHECKPOINT_DIR)/checkpoint_$(RUN_ID).json"
	@echo ""

eval-resume:
	@echo "Resuming ablation study..."
	@echo "Run ID: $(RUN_ID)"
	@echo ""
	@if not exist "$(CHECKPOINT_DIR)\checkpoint_$(RUN_ID).json" (echo ERROR: Checkpoint not found! && exit 1)
	$(PY) scripts/run_ablation_study.py \
		--questions data/eval/questions_$(NUM_QUESTIONS).jsonl \
		--output $(EVAL_OUTPUT) \
		--checkpoint-dir $(CHECKPOINT_DIR) \
		--run-id $(RUN_ID) \
		--resume \
		--api-url http://localhost:8080
	@echo ""
	@echo "Ablation study complete!"
	@echo ""

eval-quick:
	@echo "Running QUICK test (10 questions, 3 configs)..."
	@echo "This should take ~2-5 minutes"
	@echo ""
	@if not exist "results" mkdir results
	$(PY) scripts/run_ablation_study.py \
		--questions data/eval/questions_$(NUM_QUESTIONS).jsonl \
		--output results/quick_test.json \
		--configs baseline full_no_cove full_cove_auto \
		--max-questions 10 \
		--api-url http://localhost:8080
	@echo ""
	@echo "Quick test complete!"
	@echo "Results: results/quick_test.json"
	@echo ""

eval-clean:
	@echo "Cleaning evaluation data..."
	@if exist "$(CHECKPOINT_DIR)" rmdir /s /q $(CHECKPOINT_DIR)
	@if exist "results" rmdir /s /q results
	@if exist "data\eval" rmdir /s /q data\eval
	@echo "Evaluation data cleaned!"
	@echo ""