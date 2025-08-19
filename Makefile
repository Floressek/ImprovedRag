PY=python
PIP=pip

.PHONY: install install-dev pre-commit fmt lint type test ingest index api eval up down ci-clean

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

pre-commit:
	pre-commit run --all-files

fmt:
	ruff check --fix .
	ruff format .
	black .
	isort .

lint:
	ruff check .
	black --check .
	isort --check-only .

type:
	mypy src/ragx

test:
	pytest -q

ingest:
	$(PY) scripts/ingest_wiki.py --config configs/models.yaml

index:
	$(PY) scripts/build_index.py --config configs/models.yaml

api:
	uvicorn ragx.api.main:app --host 0.0.0.0 --port 8000 --reload

eval:
	$(PY) scripts/run_eval.py --config configs/eval.yaml

up:
	docker compose up -d --build

down:
	docker compose down

ci-clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov coverage.xml

ingest-wiki:
	python -m ragx.cli.main ingest wikipedia --language en --max-docs 10000

build-index:
	python -m ragx.cli.main index build --config configs/models.yaml

start-qdrant:
	docker-compose up -d qdrant

api-dev:
	uvicorn ragx.api.main:app --host 0.0.0.0 --port 8000 --reload

full-setup: start-qdrant ingest-wiki build-index
	@echo "RAG system ready!"
