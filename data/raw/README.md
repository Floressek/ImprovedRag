# 📦 Raw Wikipedia Dump (Polish)

This directory contains the **unprocessed source dump** of the Polish Wikipedia used for
vectorization and ingestion into the RAGX knowledge base.

## Contents

- `pl_wiki_dump/`
    - `plwiki-20250601-pages-articles-multistream.xml.bz2` – official Polish Wikipedia dump
      downloaded from the Wikimedia Dumps portal:
      [https://dumps.wikimedia.org/plwiki/latest/](https://dumps.wikimedia.org/plwiki/latest/)
    - `.gitkeep` – placeholder for Git tracking when dump files are not included.

> ⚠️ Full dump archives are not stored in this repository due to GitHub’s file size
> limits. To reproduce the ingestion, download the original dump manually or via the
> `Makefile` provided in the root project.

## Metadata

- **Source:** [Polish Wikipedia](https://pl.wikipedia.org/)
- **Snapshot date:** 2025-06-01
- **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- **Processing tool:** `wikiextractor` (Dockerized)

## Notes

> This data serves as the initial raw text corpus. It is not directly used by Qdrant but
> is required for preprocessing and document chunking steps.