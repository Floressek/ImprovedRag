# Makefile shortcuts (placeholder)
.PHONY: ingest index api test

ingest:
	python scripts\ingest_wiki.py

index:
	python scripts\build_index.py

api:
	python scripts\serve_api.py

 test:
	python -m pytest -q
