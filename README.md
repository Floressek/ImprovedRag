# RAGx - Advanced Retrieval-Augmented Generation System

Ablation oriented RAG system featuring adaptive query rewriting, multihop retrieval, Chain-of-Verification (CoVe), and cross-encoder reranking. Designed for Polish and multilingual knowledge bases.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Experimental Results](#experimental-results)
- [Project Structure](#project-structure)
- [License](#license)
- [References](#references)

---

## Overview

RAGx is an advanced Retrieval-Augmented Generation system that combines semantic search with large language models to provide accurate, citation-backed answers. The system features:

- **Adaptive Query Processing** - Automatic detection of query complexity and decomposition into sub-queries
- **Multihop Retrieval** - Three-stage reranking pipeline for complex questions requiring information synthesis
- **Chain-of-Verification (CoVe)** - Post-generation claim verification and correction
- **Multi-Provider LLM Support** - HuggingFace, Ollama, vLLM, and OpenAI-compatible APIs

The system is optimized for Polish Wikipedia but supports any multilingual corpus through configurable embeddings.

---

## Features

### Core RAG Pipeline

| Component | Description |
|-----------|-------------|
| Semantic Retrieval | Multilingual embeddings (GTE, E5, BGE) with Qdrant HNSW indexing |
| Cross-Encoder Reranking | Jina Reranker v2 for precision improvement on top-K results |
| LLM Generation | Multi-provider support with 4-bit quantization |
| Citation Enforcement | Inline source citations `[N]` for every factual claim |

### Query Processing

| Feature | Description |
|---------|-------------|
| Linguistic Analysis | spaCy-based POS tagging, dependency parsing, NER |
| Query Type Detection | Automatic classification: comparison, verification, similarity, chaining, temporal, aggregation, superlative |
| Sub-Query Decomposition | LLM-powered decomposition of complex questions |
| Adaptive Rewriting | Query expansion and reformulation based on linguistic features |

### Retrieval Strategies

| Mode | Description |
|------|-------------|
| Single Query | Standard retrieval with optional reranking |
| Multihop | Parallel retrieval for sub-queries with three-stage fusion (local, fusion, global) |

### Verification (CoVe)

| Component | Description |
|-----------|-------------|
| Claim Extraction | Automatic extraction of verifiable claims from generated answers |
| NLI Verification | Natural Language Inference-based claim verification against evidence |
| Correction | Automatic correction of unsupported or contradicted claims |
| Citation Injection | Adding citations for verified claims |

---

## Architecture

```
User Query
    |
    v
+------------------------------------------+
|  1. LINGUISTIC ANALYSIS (spaCy)          |
|  - POS tagging, dependency parsing       |
|  - Entity extraction, clause counting    |
+------------------------------------------+
    |
    v
+------------------------------------------+
|  2. ADAPTIVE QUERY REWRITING (LLM)       |
|  - Query type detection                  |
|  - Decomposition decision                |
|  - Sub-query generation                  |
+------------------------------------------+
    |
    +---------------+---------------+
    |               |               |
 [Simple]       [Multihop]
    |               |
    v               v
+-------------+  +------------------------+
| Single      |  | Parallel Retrieval     |
| Retrieval   |  | (per sub-query)        |
+-------------+  +------------------------+
    |               |
    v               v
+-------------+  +------------------------+
| Standard    |  | Three-Stage Reranking: |
| Reranking   |  | 1. Local (per query)   |
|             |  | 2. Fusion (by doc_id)  |
|             |  | 3. Global (original Q) |
+-------------+  +------------------------+
    |               |
    +-------+-------+
            |
            v
+------------------------------------------+
|  3. PROMPT ENGINEERING                   |
|  - Template selection (basic/enhanced)   |
|  - Context formatting with metadata      |
|  - Language detection                    |
+------------------------------------------+
    |
    v
+------------------------------------------+
|  4. LLM GENERATION                       |
|  - Multi-provider (HF/Ollama/vLLM/API)   |
|  - Chain-of-Thought reasoning            |
|  - Citation formatting                   |
+------------------------------------------+
    |
    v
+------------------------------------------+
|  5. CHAIN-OF-VERIFICATION (CoVe)         |
|  - Claim extraction                      |
|  - NLI verification                      |
|  - Correction and citation injection     |
+------------------------------------------+
    |
    v
Final Answer + Sources + Metadata
```

---

## Requirements

### Development Environment

| Resource | Specification                              |
|----------|--------------------------------------------|
| GPU | NVIDIA RTX 4070 (12GB VRAM)                |
| CPU | AMD Ryzen 7 7800X3D (8 cores / 16 threads) |
| RAM | 32GB DDR5                                  |
| Storage | 1TB                                        |
| OS | Windows 11 Home                            |
| LLM | Qwen2.5-14B / Qwen3-8B (4-bit via Ollama)  |

### Production Environment (curtesy of Military University of Technology Cloud Laboratory)

| Resource | Specification |
|----------|---------------|
| GPU | NVIDIA H100 (96GB VRAM) |
| CPU | AMD Ryzen 7 7800X3D (8 cores / 16 threads) |
| RAM | 128GB DDR5 |
| Storage | 2TB |
| OS | Ubuntu 22.04 LTS |
| LLM | Qwen3-32B (4-bit via vLLM) |

### Shared Configuration

| Component | Model |
|-----------|-------|
| Embedding | Alibaba-NLP/gte-multilingual-base |
| Reranker | jinaai/jina-reranker-v2-base-multilingual |
| Linguistic Analysis | spaCy pl_core_news_md |
| Vector Store | Qdrant (self-hosted) |

### Software Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| Docker | 20.10+ |
| CUDA | 12.0+ (optional) |

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/floressek/ragx.git
cd ragx
```

### 2. Install Dependencies

Using uv (recommended):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install --system -e .
```

Using pip:
```bash
pip install -e .
```

Using make:
```bash
make install
```

### 3. Install spaCy Models

```bash
python -m spacy download pl_core_news_md
python -m spacy download en_core_web_sm
```

### 4. Start Qdrant

```bash
docker-compose up -d qdrant
```

### 5. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

---

## Configuration

### Environment Variables

```bash
# Vector Store
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=ragx_documents_v3

# Embeddings
EMBEDDING_MODEL=Alibaba-NLP/gte-multilingual-base
EMBEDDING_BATCH_SIZE=64
EMBEDDING_USE_PREFIXES=true

# Reranking
RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
RERANKER_BATCH_SIZE=16

# LLM Provider: huggingface | ollama | vllm | api
LLM_PROVIDER=huggingface
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_LOAD_IN_4BIT=true
LLM_TEMPERATURE=0.2
LLM_MAX_NEW_TOKENS=2000

# Ollama (alternative)
# LLM_PROVIDER=ollama
# OLLAMA_HOST=http://localhost:11434
# LLM_MODEL_NAME_OLLAMA=qwen3:4b

# vLLM (production)
# LLM_PROVIDER=vllm
# LLM_API_BASE_URL=http://localhost:8000/v1
# LLM_API_MODEL_NAME=Qwen/Qwen3-32B

# Query Rewriting
REWRITE_ENABLED=true
REWRITE_TEMPERATURE=0.2
REWRITE_MAX_TOKENS=4096

# Multihop Configuration
MULTIHOP_FUSION_STRATEGY=max
MULTIHOP_GLOBAL_RANKER_WEIGHT=0.6
MULTIHOP_TOP_K_PER_SUBQUERY=20
MULTIHOP_FINAL_TOP_K=10

# Retrieval Pipeline
TOP_K_RETRIEVE=100
RERANK_TOP_M=80
CONTEXT_TOP_N=8

# Chain-of-Verification
COVE_ENABLED=true
COVE_USE_BATCH_NLI=true
```

### Configuration Files

| File | Purpose |
|------|---------|
| `configs/models.yaml` | Model configurations (embedder, reranker, LLM) |
| `configs/app.yaml` | Application settings |
| `configs/vector_store.qdrant.yaml` | Qdrant connection settings |
| `configs/eval.yaml` | Evaluation settings |

---

## Usage

### Data Ingestion

```bash
# Download Polish Wikipedia dump
make download-wiki

# Extract articles
make extract-wiki

# Ingest to Qdrant (test - 1k articles)
make ingest-test

# Ingest full corpus (200k+ articles)
make ingest-full

# Custom ingestion
make ingest-custom MAX_ARTICLES=50000
```

Pre-built Qdrant snapshot available at:
https://huggingface.co/datasets/Floressek/wiki-1m-qdrant-snapshot

### API Server

```bash
# Start FastAPI server
make api

# Or with auto-reload for development
make api-dev

# Manual start
python -m uvicorn src.ragx.api.main:app --host 0.0.0.0 --port 8000
```

### Interactive Chat UI

```bash
./launch_ui.sh

# Or directly
streamlit run src/ragx/ui/chat_app.py
```

### Command Line

```bash
# Search
make search QUERY="sztuczna inteligencja"

# Check status
make status
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api` | API information and available endpoints |
| GET | `/info/health` | Health check with model status |
| POST | `/ask/baseline` | Simple RAG pipeline (retrieval + LLM) |
| POST | `/ask/enhanced` | Full pipeline with query rewriting and multihop |
| POST | `/llm/generate` | Direct LLM access (no RAG) |
| POST | `/search/search` | Vector search only |
| POST | `/search/rerank` | Search with reranking |
| POST | `/analysis/linguistic` | Linguistic analysis of query |
| POST | `/analysis/rewrite` | Query rewriting analysis |
| POST | `/cove/verify` | CoVe verification of answer |
| POST | `/eval/ablation` | Ablation study endpoint with configurable toggles |

### Example Requests

**Baseline Pipeline:**
```bash
curl -X POST "http://localhost:8000/ask/baseline" \
  -H "Content-Type: application/json" \
  -d '{"query": "Co to jest sztuczna inteligencja?", "top_k": 5}'
```

**Enhanced Pipeline (with query rewriting and multihop):**
```bash
curl -X POST "http://localhost:8000/ask/enhanced" \
  -H "Content-Type: application/json" \
  -d '{"query": "ziemniaki vs pomidory, co ma wiecej blonnika?"}'
```

**Ablation Study:**
```bash
curl -X POST "http://localhost:8000/eval/ablation" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Kto zalozyl Krakow?",
    "top_k": 8,
    "query_analysis_enabled": true,
    "reranker_enabled": true,
    "cot_enabled": true,
    "cove_mode": "auto",
    "prompt_template": "auto"
  }'
```

### Response Format

```json
{
  "answer": "Answer text with citations [1][2]...",
  "sources": [
    {
      "id": "doc_id",
      "text": "Source text...",
      "doc_title": "Document Title",
      "retrieval_score": 0.85,
      "rerank_score": 0.92
    }
  ],
  "metadata": {
    "pipeline": "enhanced",
    "is_multihop": true,
    "sub_queries": ["sub-query 1", "sub-query 2"],
    "query_type": "comparison",
    "rewrite_time_ms": 450.2,
    "retrieval_time_ms": 25.8,
    "rerank_time_ms": 180.5,
    "llm_time_ms": 920.1,
    "cove_time_ms": 340.0,
    "total_time_ms": 1916.6
  }
}
```

---

## Evaluation

### Methodology

Evaluation was conducted using ablation experiments on a test set of **1000 synthetic questions** generated from the Polish Wikipedia corpus (1M articles). The evaluation uses **RAGAS metrics** and was performed on infrastructure provided by the Military University of Technology Cloud Laboratory.

#### Why Synthetic Questions?

Unlike standard benchmarks (PolQA, MKQA), synthetic questions generated directly from the indexed corpus provide:
- Full control over grounding - each question has exact source documents in the Qdrant database
- Coverage of all query types supported by the system
- Polish language optimization

### Test Set Generation

The `WikipediaQuestionGenerator` creates evaluation questions through the following pipeline:

1. **Article Sampling** - Random selection of article chunks from ingested data (minimum 200 characters)
2. **Question Generation** - LLM-based generation (Qwen3:32B) with dedicated prompts per question type
3. **Grounding Validation** - Cross-encoder verification that "ground truth" exists in the source article
4. **Export** - JSONL format with fields: `ground_truth`, `type`, `contexts`

**Example generated question (JSONL format):**
```json
{
  "question": "Jak nazywa sie wspolnik do Neville'a Roundego?",
  "ground_truth": "Tekstor zostal przez Aldine...",
  "type": "simple",
  "source_title": "Neville'a...",
  "contexts": [
    "https://pl.wikipedia.org/wiki/Aldine#12661"
  ]
}
```

### RAGAS Metrics

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Factual consistency of answer with retrieved contexts |
| **Answer Relevancy** | Relevance of answer to the question |
| **Context Precision** | Proportion of relevant contexts in retrieved set |
| **Context Recall** | Coverage of ground truth by retrieved contexts |

### Running Evaluation

```bash
# Generate test questions (default: 1000)
make eval-generate NUM_QUESTIONS=1000

# Start RAG API server
make eval-api

# Run ablation study with checkpointing
make eval-run

# Resume interrupted evaluation
make eval-resume RUN_ID=study_20240115_143022

# Quick validation (10 questions, 3 configs)
make eval-quick

# Clean checkpoints and results
make eval-clean
```

---

## Experimental Results

### Ablation Study Results

Full results across 12 configurations on 1000 test questions. Bold values indicate best performance for each metric.

| Configuration | Faithfulness | Relevancy | Precision | Recall | Latency | Cov |
|--------------|--------------|-----------|-----------|--------|---------|-----|
| baseline | 0.768 | 0.594 | 0.463 | 0.600 | 2.7s | 0.00 |
| enhanced_only | 0.850 | 0.646 | 0.443 | 0.622 | 7.6s | 0.00 |
| cot_only | 0.884 | 0.641 | 0.440 | 0.614 | 9.6s | 0.00 |
| reranker_only | 0.838 | 0.680 | 0.501 | 0.698 | 3.9s | 0.00 |
| cove_auto_only | 0.872 | 0.621 | 0.448 | 0.613 | 44.3s | 0.00 |
| cot_enhanced | 0.823 | 0.653 | 0.431 | 0.610 | 12.7s | 0.00 |
| multihop_only | **0.891** | 0.714 | 0.494 | **0.829** | 18.4s | 0.64 |
| multihop+cot | 0.870 | **0.762** | 0.493 | 0.828 | 25.7s | 0.62 |
| full_no_cove | 0.881 | 0.721 | 0.506 | 0.823 | 24.5s | 0.61 |
| full_cove_auto | 0.855 | 0.732 | 0.516 | 0.810 | 62.8s | 0.64 |
| full_cove_metadata | 0.858 | 0.743 | 0.498 | 0.827 | 60.4s | 0.62 |
| full_cove_suggest | 0.832 | 0.756 | **0.522** | 0.810 | 63.6s | 0.60 |

### Component Impact Analysis

Individual component contribution compared to baseline:

| Component | Faithfulness | Relevancy | Recall | Latency |
|-----------|--------------|-----------|--------|---------|
| Reranker | +9.1% (0.77->0.84) | +15.3% (0.59->0.68) | +16.7% (0.60->0.70) | +1.2s |
| CoT | +14.2% (0.77->0.88) | +8.5% (0.59->0.64) | +1.6% (0.60->0.61) | +6.9s |
| Multihop | +15.6% (0.77->0.89) | +20.3% (0.59->0.71) | +38.4% (0.60->0.83) | +15.7s |
| CoVe (auto) | +11.6% (0.77->0.86) | +5.1% (0.59->0.62) | +1.6% (0.60->0.61) | +41.6s |

### Key Findings

1. **Multihop module dominates** in Faithfulness (0.891) and Context Recall (0.829)
2. **Multihop + CoT combination** achieves highest Answer Relevancy (0.762)
3. **Baseline performs lowest** across all metrics
4. **Any component addition** improves results over baseline
5. **Best ROI**: Multihop provides +38.4% Recall improvement with acceptable latency cost

### Multihop Classification Distribution

62-64% of test queries are classified as multihop, indicating significant proportion of complex questions in the test set:

| Configuration | Multihop | Simple | Coverage |
|--------------|----------|--------|----------|
| multihop_only | 645 | 355 | 64.0% |
| multihop+cot | 620 | 380 | 62.0% |
| full_no_cove | 615 | 385 | 61.0% |
| full_cove_auto | 640 | 360 | 63.5% |
| full_cove_metadata | 630 | 370 | 61.5% |
| full_cove_suggest | 620 | 380 | 62.0% |

### Quality vs Latency Trade-off

| Tier | Configurations | Latency | Characteristics |
|------|----------------|---------|-----------------|
| **Fast** | baseline, reranker_only | 2.7-3.9s | Production real-time, basic quality |
| **Medium** | enhanced, cot_only, cot_enhanced | 7.6-12.7s | Quality/speed balance |
| **Slow** | multihop_only, multihop+cot, full_no_cove | 18.4-25.7s | High quality, acceptable latency |
| **Very Slow** | cove_auto_only, full_cove_* | 44.3-63.6s | Highest quality, offline use |

### Recommended Configurations

| Use Case | Configuration | Expected Latency |
|----------|---------------|------------------|
| Real-time chat | reranker_only | ~4s |
| Balanced production | multihop_only | ~18s |
| Maximum quality (async) | full_cove_metadata | ~60s |

---

## Project Structure

```
ragx/
├── src/ragx/
│   ├── api/                    # FastAPI server
│   │   ├── routers/            # Endpoint handlers
│   │   ├── schemas/            # Pydantic models
│   │   ├── dependencies.py     # Dependency injection
│   │   └── main.py             # Application entry point
│   │
│   ├── ingestion/              # Data ingestion pipeline
│   │   ├── chunkers/           # Text chunking strategies
│   │   ├── pipelines/          # Ingestion orchestration
│   │   ├── extractions/        # Wikipedia extraction
│   │   └── utils/              # Ingestion utilities
│   │
│   ├── retrieval/              # Retrieval components
│   │   ├── embedder/           # Bi-encoder embeddings
│   │   ├── rerankers/          # Cross-encoder reranking
│   │   ├── analyzers/          # Linguistic analysis
│   │   ├── rewriters/          # Query rewriting
│   │   ├── cove/               # Chain-of-Verification
│   │   └── vector_stores/      # Qdrant integration
│   │
│   ├── pipelines/              # RAG pipelines
│   │   ├── base.py             # Abstract base
│   │   ├── baseline.py         # Simple RAG
│   │   ├── enhanced.py         # Full pipeline
│   │   └── enhancers/          # Pipeline enhancers
│   │
│   ├── generation/             # LLM generation
│   │   ├── inference.py        # Multi-provider inference
│   │   ├── providers/          # Provider implementations
│   │   └── prompts/            # Prompt templates
│   │
│   ├── ui/                     # Streamlit chat interface -> claude generated, beta testing.
│   │   ├── chat_app.py         # Main application
│   │   ├── components/         # UI components
│   │   └── config/             # UI configuration
│   │
│   └── utils/                  # Shared utilities
│       ├── settings.py         # Configuration management
│       ├── logging_config.py   # Logging setup
│       └── model_registry.py   # Model caching
│
├── configs/                    # Configuration files
├── data/                       # Data directory
│   ├── raw/                    # Raw Wikipedia dumps
│   ├── processed/              # Extracted articles
│   └── db_snapshots/           # Qdrant snapshots
├── scripts/                    # Utility scripts
├── results/                    # Evaluation results
├── docker-compose.yml          # Docker services
├── Makefile                    # Build commands
├── pyproject.toml              # Project dependencies
└── README.md
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

### Papers

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
- [Query Rewriting for Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2305.14283) (Jagerman et al., 2023)
- [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600) (Yang et al., 2018)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019)
- [Chain-of-Verification Reduces Hallucination in Large Language Models](https://arxiv.org/abs/2309.11495) (Dhuliawala et al., 2023)

### Libraries

- [Sentence-Transformers](https://www.sbert.net/)
- [Qdrant](https://qdrant.tech/)
- [LlamaIndex](https://www.llamaindex.ai/)
- [spaCy](https://spacy.io/)
- [Transformers](https://huggingface.co/transformers/)
- [FastAPI](https://fastapi.tiangolo.com/)

### Models

- GTE-multilingual (Alibaba)
- Jina Reranker v2 (Jina AI)
- Qwen2.5 / Qwen3 (Alibaba Cloud)
