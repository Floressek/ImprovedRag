"""Wikipedia extraction → chunking (placeholder script)."""
from __future__ import annotations
from src.ragx.ingestion.pipeline import run_ingestion


def main() -> None:
    chunks = run_ingestion("data/raw/wiki.xml")
    print(f"Ingested {len(chunks)} chunks")


if __name__ == "__main__":
    main()
