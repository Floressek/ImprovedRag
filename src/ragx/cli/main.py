"""Typer CLI entry point (placeholder without Typer dependency)."""
from __future__ import annotations
from .commands import ingest, index, query, eval as eval_cmd, serve


def main() -> None:
    """Very small dispatcher mimic (no CLI parsing)."""
    print("ragx CLI placeholder. Available commands: ingest, index, query, eval, serve")
    # Not implementing actual CLI parsing to avoid dependencies.


if __name__ == "__main__":
    main()
