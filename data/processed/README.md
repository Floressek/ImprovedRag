# 🧠 Processed Wikipedia Data

This directory contains the **cleaned and tokenized Wikipedia corpus**, extracted from
the raw XML dump using `WikiExtractor`. The result is a structured dataset ready for
embedding generation and ingestion into the Qdrant vector database.

## Structure

```
processed/
└── wiki_extracted/
├── AA/
│ ├── wiki_00.jsonl
│ ├── wiki_01.jsonl
│ └── ...
├── AB/
└── ...
```

Each `.jsonl` file contains JSON-formatted articles with the following fields:

```json
{
  "id": "12345",
  "title": "Przykład artykułu",
  "text": "Treść artykułu po ekstrakcji, bez znaczników HTML...",
  "url": "https://pl.wikipedia.org/wiki/Przykład_artykułu"
}
```

## Generation

> ⚠️ The data was extracted automatically in a Docker container.

Resulting files are indexed and consumed by the embedding pipeline (src/ragx/ingestion)
to produce Qdrant vectors.

## License

- **Source:** [Polish Wikipedia](https://pl.wikipedia.org/)
- **Snapshot date:** 2025-06-01
- **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- **Processing tool:** `wikiextractor` (Dockerized)

> *Transformation: Text cleaned and chunked; metadata preserved.*

> ⚠️ **The processed files remain under the same CC BY-SA license as the source material.**
