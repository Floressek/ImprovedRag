# 🚀 RAGx - Advanced RAG System with Query Rewriting & Multihop Retrieval

Retrieval-Augmented Generation (RAG) system with advanced query processing including **Linguistic Analysis**, **Adaptive Query Rewriting**, **Multihop Retrieval**, and **Cross-Encoder Reranking**.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#️-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Performance](#-performance)
- [Contributing](#-contributing)

---

## ✨ Features

### Core RAG Pipeline
- **🔍 Semantic Retrieval** - Multi-lingual embeddings (GTE, E5, BGE)
- **⚡ Vector Store** - Qdrant with HNSW indexing for fast similarity search
- **🎯 Cross-Encoder Reranking** - Improves precision by re-scoring top-K results
- **🤖 Multi-Provider LLM** - HuggingFace, Ollama, vLLM, API (LM Studio, OpenAI-compatible)

### Advanced Query Processing 🆕
- **🧠 Linguistic Analysis** - spaCy-based POS tagging, dependency parsing, NER
- **🔄 Adaptive Query Rewriting** - LLM-powered query decomposition and expansion
- **🎯 Query Type Detection** - Automatic detection of:
    - **Verification** - "Is X the largest...?"
    - **Comparison** - "X vs Y in terms of Z"
    - **Similarity** - "What do X and Y have in common?"
    - **Chaining** - "Who directed the movie starring X?"
    - **Temporal** - "Events between X and Y"
    - **Aggregation** - "How many X..."
    - **Superlative** - "What's the best X under Y?"
- **📊 Sub-Query Decomposition** - Intelligent breaking down of complex questions
- **🔗 Multihop Retrieval** - Three-stage reranking (local → fusion → global)

### Advanced Methods
- **✅ Citation Enforcement** - Inline source citations `[N]` for every claim
- **📝 Advanced Prompting** - Template system with CoT support and language detection
- **🧩 Semantic Chunking** - Context-aware text splitting with LlamaIndex
- **🎨 Query-Type-Specific Fusion** - Adaptive weights based on query complexity

### Production Features
- **📊 Progress Tracking** - Resume ingestion from where you left off
- **🔄 Incremental Indexing** - Skip already processed files
- **⚙️ Configurable Pipeline** - YAML + .env for easy configuration
- **🌐 REST API** - FastAPI server with multiple endpoints
- **📈 Performance Monitoring** - Built-in metrics and logging

---

## 🏗️ Architecture

### Enhanced Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Linguistic Analysis (spaCy)                        │
│  - POS tagging, dependency parsing, NER                     │
│  - Syntax depth, clause counting                            │
│  - Entity extraction                                        │
│  Output: LinguisticFeatures                                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Adaptive Query Rewriting (LLM)                     │
│  - Query type detection (comparison, similarity, etc.)      │
│  - Decision: decompose / expand / passthrough               │
│  - Multi-hop decomposition into sub-queries                 │
│  Output: is_multihop, sub_queries[], query_type            │
└───────────────────────┬─────────────────────────────────────┘
                        │
              ┌─────────┴─────────┐
              │                   │
         [Simple]            [Multihop]
              │                   │
              ▼                   ▼
   ┌──────────────────┐  ┌──────────────────────────┐
   │ Single Query     │  │ Multiple Sub-Queries     │
   │ Retrieval        │  │ Parallel Retrieval       │
   └─────┬────────────┘  └────────┬─────────────────┘
         │                         │
         ▼                         ▼
   ┌──────────────────┐  ┌──────────────────────────┐
   │ Standard         │  │ Three-Stage Reranking:   │
   │ Reranking        │  │ 1. Local (per subquery)  │
   │ (Cross-Encoder)  │  │ 2. Fusion (by doc_id)    │
   │                  │  │ 3. Global (original Q)   │
   └─────┬────────────┘  └────────┬─────────────────┘
         │                         │
         └────────┬────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Prompt Engineering                                 │
│  - Template selection (basic/enhanced/multihop)             │
│  - Context formatting with metadata                         │
│  - Language detection & CoT injection                       │
│  - Citation instructions                                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: LLM Generation                                      │
│  - Multi-provider support (HF/Ollama/vLLM/API)              │
│  - Temperature control (0.2-0.7)                            │
│  - Chain-of-Thought reasoning                               │
│  Output: Answer with inline citations [N]                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Final Answer to User                        │
│  + Source documents with scores                             │
│  + Metadata (timings, query_type, sub_queries)             │
└─────────────────────────────────────────────────────────────┘
```

### Query Type Examples

**Comparison Query:**
```
User: "ziemniaki vs pomidory, co ma więcej błonnika?"
└─> Detected: comparison
└─> Sub-queries:
    1. "Ile błonnika mają ziemniaki?"
    2. "Ile błonnika mają pomidory?"
└─> Fusion: MAX strategy
└─> Answer: "Ziemniaki zawierają około 2.2g błonnika na 100g [1], 
            podczas gdy pomidory około 1.2g [2]."
```

**Verification Query:**
```
User: "Polska to największy kraj europejski?"
└─> Detected: verification
└─> Sub-queries:
    1. "Jaka jest powierzchnia Polski?"
    2. "Jaki jest największy kraj w Europie?"
└─> Fusion: Query-type weight = 0.2 (trust local more)
└─> Answer: "Nie, Polska nie jest największym krajem w Europie..."
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.12+**
- **Docker** (for Qdrant)
- **20GB+ RAM** (32GB recommended)
- **CUDA GPU** (optional, for faster processing)

### 1. Clone & Install

```bash
git clone https://github.com/floressek/ragx.git
cd ragx

# Install dependencies
make install

# Or manually:
pip install -e .
```

### 2. Start Qdrant

```bash
make setup-qdrant
# Or manually:
docker-compose up -d qdrant
```

> Dataset used for qdrant:
https://huggingface.co/datasets/Floressek/wiki-1m-qdrant-snapshot

### 3. Configure

```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings
# Key variables:
# - EMBEDDING_MODEL
# - QDRANT_COLLECTION
# - CHUNK_SIZE, CHUNK_OVERLAP
# - REWRITE_ENABLED=true          # Enable query rewriting
# - LLM_PROVIDER=huggingface      # or ollama, vllm, api
```

### 4. Ingest Data

```bash
# Download Polish Wikipedia (small chunk for testing)
make download-wiki

# Extract articles
make extract-wiki

# Index into Qdrant (1k articles for testing)
make ingest-test

# Or full ingestion (200k articles):
# make ingest-full
```

### 5. Start API Server

```bash
# Start FastAPI server
make api

# Or manually:
python -m uvicorn src.ragx.api.main:app --host 0.0.0.0 --port 8000
```

### 6. Try It Out!

```bash
# Simple search
curl -X POST "http://localhost:8000/ask/baseline" \
  -H "Content-Type: application/json" \
  -d '{"query": "sztuczna inteligencja"}'

# Enhanced pipeline with query rewriting
curl -X POST "http://localhost:8000/ask/enhanced" \
  -H "Content-Type: application/json" \
  -d '{"query": "ziemniaki vs pomidory błonnik"}'

# Linguistic analysis
curl -X POST "http://localhost:8000/analysis/linguistic" \
  -H "Content-Type: application/json" \
  -d '{"query": "Co łączy mitologię słowiańską i nordycką?"}'
```

---

## 💻 Installation

### Option 1: Using `uv` (Recommended - Fast!)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project
uv pip install --system -e .
```

### Option 2: Using `pip`

```bash
pip install --upgrade pip
pip install -e .
```

### Option 3: Using `make`

```bash
make install
```

### Dependencies

Core libraries:
- `sentence-transformers` - Embeddings & reranking
- `qdrant-client` - Vector database
- `transformers` - LLM inference
- `llama-index` - Semantic chunking
- `spacy` - Linguistic analysis
- `fastapi` - REST API
- `pyyaml` - Configuration

### spaCy Models

```bash
# Polish language model (recommended)
python -m spacy download pl_core_news_md

# English fallback
python -m spacy download en_core_web_sm
```

---

## ⚙️ Configuration

### Environment Variables (.env)

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

# LLM Provider (huggingface, ollama, vllm, api)
LLM_PROVIDER=huggingface
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_LOAD_IN_4BIT=true
LLM_MAX_NEW_TOKENS=2000
LLM_TEMPERATURE=0.2

# Alternative: Ollama
# LLM_PROVIDER=ollama
# OLLAMA_HOST=http://localhost:11434
# LLM_MODEL_NAME_OLLAMA=qwen3:4b

# Alternative: API (LM Studio, OpenAI-compatible)
# LLM_PROVIDER=api
# LLM_API_BASE_URL=http://localhost:1234/v1
# LLM_API_MODEL_NAME=local-model
# LLM_API_KEY=your-key

# Query Rewriting 🆕
REWRITE_ENABLED=true
REWRITE_TEMPERATURE=0.2
REWRITE_MAX_TOKENS=4096
REWRITE_VERIFY_BEFORE_RETRIEVAL=false

# Multihop Configuration 🆕
MULTIHOP_FUSION_STRATEGY=max           # max, mean, weighted_mean
MULTIHOP_GLOBAL_RANKER_WEIGHT=0.6     # 0.0-1.0
MULTIHOP_TOP_K_PER_SUBQUERY=20
MULTIHOP_FINAL_TOP_K=10

# Retrieval Pipeline
TOP_K_RETRIEVE=100     # Initial retrieval
RERANK_TOP_M=80        # Candidates for reranking
CONTEXT_TOP_N=8        # Final chunks to LLM

# Chunking
CHUNKER_STRATEGY=semantic
CHUNK_SIZE=512
CHUNK_OVERLAP=96
```

---

## 📖 Usage

### Command-Line Interface

```bash
# Ingestion pipelines
python -m src.ragx.ingestion.pipelines --help

# Available commands:
python -m src.ragx.ingestion.pipelines download --language pl
python -m src.ragx.ingestion.pipelines ingest <source> --max-articles 10000
python -m src.ragx.ingestion.pipelines status
python -m src.ragx.ingestion.pipelines search "query text"
```

### Makefile Commands

```bash
# Setup
make install              # Install dependencies
make setup-qdrant         # Start Qdrant container

# Wikipedia Pipeline
make download-wiki        # Download PL Wikipedia dump
make extract-wiki         # Extract articles to JSON
make ingest-test          # Test ingestion (1k articles)
make ingest-full          # Full ingestion (200k articles)

# Progress Tracking
make ingest-resume        # Resume from last processed file
make ingest-from FILE=wiki_05  # Start from specific file
make status-detailed      # Show file-by-file history

# API Server
make api                  # Start FastAPI server
make api-dev              # Start with auto-reload

# Search & Status
make search QUERY="..."   # Search for query
make status               # Check system status

# Maintenance
make clean                # Clean cache files
make clean-data           # Clean data files
make clean-all            # Nuclear clean
```

---

## 🌐 API Documentation

### Endpoints Overview

```
GET  /api                      # API information
GET  /info/health              # Health check with model status

POST /ask/baseline             # Simple RAG pipeline
POST /ask/enhanced             # Enhanced pipeline with query rewriting

POST /llm/generate             # Direct LLM access (no RAG)

POST /search/search            # Vector search only
POST /search/rerank            # Search + reranking

POST /analysis/linguistic      # Linguistic analysis
POST /analysis/multihop        # Multihop search with detailed metadata
```

### Example Requests

#### 1. Baseline Pipeline (Simple RAG)

```bash
curl -X POST "http://localhost:8000/ask/baseline" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Co to jest sztuczna inteligencja?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "answer": "Sztuczna inteligencja (SI) to dział informatyki zajmujący się...",
  "sources": [
    {
      "id": "doc123",
      "doc_title": "Sztuczna inteligencja",
      "text": "...",
      "retrieval_score": 0.85,
      "rerank_score": null
    }
  ],
  "metadata": {
    "pipeline": "baseline",
    "retrieval_time_ms": 12.5,
    "llm_time_ms": 850.2,
    "total_time_ms": 862.7,
    "num_sources": 5
  }
}
```

#### 2. Enhanced Pipeline (Query Rewriting + Multihop)

```bash
curl -X POST "http://localhost:8000/ask/enhanced" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ziemniaki vs pomidory, co ma więcej błonnika?",
    "top_k": 8
  }'
```

**Response:**
```json
{
  "answer": "Ziemniaki zawierają około 2.2g błonnika na 100g [1][2], podczas gdy pomidory około 1.2g [3][4]...",
  "sources": [
    {
      "id": "doc456",
      "doc_title": "Ziemniaki",
      "text": "...",
      "local_rerank_score": 0.92,
      "fused_score": 0.88,
      "global_rerank_score": 0.85,
      "final_score": 0.87,
      "fusion_metadata": {
        "source_subqueries": ["Ile błonnika mają ziemniaki?"],
        "num_occurrences": 1
      }
    }
  ],
  "metadata": {
    "pipeline": "enhanced",
    "is_multihop": true,
    "sub_queries": [
      "Ile błonnika mają ziemniaki?",
      "Ile błonnika mają pomidory?"
    ],
    "query_type": "comparison",
    "reasoning": "comparison by fiber",
    "rewrite_time_ms": 450.2,
    "retrieval_time_ms": 25.8,
    "rerank_time_ms": 180.5,
    "llm_time_ms": 920.1,
    "total_time_ms": 1576.6,
    "num_candidates": 200,
    "num_sources": 8
  }
}
```

#### 3. Linguistic Analysis

```bash
curl -X POST "http://localhost:8000/analysis/linguistic" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Co łączy mitologię słowiańską i nordycką?"
  }'
```

**Response:**
```json
{
  "query": "Co łączy mitologię słowiańską i nordycką?",
  "pos_sequence": ["PRON", "VERB", "NOUN", "ADJ", "CCONJ", "ADJ"],
  "dep_tree": [
    {"dependency": "nsubj", "head": "łączy", "child": "Co"},
    {"dependency": "ROOT", "head": "łączy", "child": "łączy"}
  ],
  "entities": [
    {"text": "mitologię słowiańską", "label": "MISC"},
    {"text": "nordycką", "label": "MISC"}
  ],
  "num_tokens": 7,
  "num_clauses": 1,
  "syntax_depth": 3,
  "has_relative_clauses": false,
  "has_conjunctions": true,
  "analysis_text": "..."
}
```

#### 4. Multihop Search with Options

```bash
curl -X POST "http://localhost:8000/analysis/multihop" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "król Mieszko I czy Bolesław Chrobry miał większy wpływ?",
    "top_k": 10,
    "use_reranker": true,
    "include_linguistic_analysis": true
  }'
```

**Features:**
- `use_reranker`: Enable/disable three-stage reranking (default: true)
- `include_linguistic_analysis`: Add linguistic features to response (default: false)
- Automatic query decomposition
- Query-type-specific fusion strategies

#### 5. Direct LLM Access (No RAG)

```bash
curl -X POST "http://localhost:8000/llm/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Wyjaśnij czym jest gradient descent",
    "temperature": 0.7,
    "max_tokens": 500,
    "chain_of_thought_enabled": true
  }'
```

---

## 📁 Project Structure

```
ragx/
├── src/ragx/
│   ├── api/                        # FastAPI server
│   │   ├── routers/
│   │   │   ├── chat.py            # /ask endpoints
│   │   │   ├── analysis.py        # /analysis endpoints 🆕
│   │   │   ├── search.py          # /search endpoints
│   │   │   ├── llm.py             # /llm endpoints
│   │   │   └── health.py          # /info endpoints
│   │   ├── schemas/               # Pydantic models
│   │   └── dependencies.py        # DI container
│   │
│   ├── ingestion/                 # Data ingestion pipeline
│   │   ├── chunkers/
│   │   │   └── chunker.py         # Semantic & token-based chunking
│   │   ├── pipelines/
│   │   │   ├── ingestion_pipeline.py
│   │   │   └── ingestion_progress.py  # Progress tracking
│   │   └── extractions/
│   │       └── wiki_extractor.py  # Wikipedia extraction
│   │
│   ├── retrieval/                 # Retrieval & reranking
│   │   ├── embedder/
│   │   │   └── embedder.py        # Bi-encoder embeddings
│   │   ├── rerankers/
│   │   │   └── reranker.py        # Cross-encoder reranking
│   │   ├── analyzers/ 🆕
│   │   │   ├── linguistic_analyzer.py   # spaCy analysis
│   │   │   └── linguistic_features.py   # Feature dataclass
│   │   ├── rewriters/ 🆕
│   │   │   ├── adaptive_rewriter.py     # Query rewriting
│   │   │   ├── prompts/
│   │   │   │   └── rewriter_prompts.yaml
│   │   │   └── tools/
│   │   │       └── parse.py             # JSON validation
│   │   ├── constants/ 🆕
│   │   │   └── query_types.py           # Query type definitions
│   │   └── vector_stores/
│   │       └── qdrant_store.py    # Qdrant integration
│   │
│   ├── pipelines/ 🆕              # RAG pipelines
│   │   ├── base.py                # Abstract base
│   │   ├── baseline.py            # Simple RAG
│   │   ├── enhanced.py            # Advanced RAG
│   │   └── enhancers/
│   │       ├── reranker.py        # Standard reranking
│   │       └── multihop_reranker.py  # Multihop reranking
│   │
│   ├── generation/                # LLM & prompting
│   │   ├── model.py               # LLM loading (HuggingFace)
│   │   ├── inference.py           # Multi-provider inference
│   │   ├── providers/
│   │   │   ├── ollama_provider.py
│   │   │   ├── vllm_provider.py
│   │   │   └── api_provider.py
│   │   └── prompts/
│   │       ├── builder.py         # Prompt templates
│   │       └── templates/
│   │           ├── basic.yaml
│   │           ├── enhanced.yaml
│   │           └── multihop.yaml  🆕
│   │
│   └── utils/
│       ├── settings.py            # Configuration management
│       ├── logging_config.py      # Structured logging
│       └── model_registry.py      # Model caching
│
├── configs/
│   ├── models.yaml                # Model configurations
│   └── app.yaml                   # App settings
│
├── data/
│   ├── raw/                       # Raw Wikipedia dumps
│   ├── processed/                 # Extracted articles
│   └── .ingestion_progress.json  # Progress tracking
│
├── tests/                         # Unit & integration tests
├── docs/                          # Additional documentation
├── docker-compose.yml             # Qdrant service
├── Makefile                       # Development commands
├── pyproject.toml                 # Project dependencies
└── README.md                      # This file
```

---

## 📊 Performance

### Benchmarks (Single GPU - RTX 4070)

| Operation | Speed | Notes |
|-----------|-------|-------|
| Embedding (batch=64) | ~1200 docs/s | GTE-multilingual-base |
| Reranking (batch=16) | ~80 pairs/s | Jina-reranker-v2 |
| Chunking (semantic) | ~50 docs/s | LlamaIndex splitter |
| LLM Generation (HF) | ~25 tokens/s | Qwen2.5-7B (4-bit) |
| LLM Generation (Ollama) | ~40 tokens/s | Qwen3:4b |
| Vector Search | <10ms | Qdrant HNSW (100k docs) |
| Query Rewriting | ~450ms | LLM-based decomposition |
| Linguistic Analysis | ~50ms | spaCy Polish model |

### Multihop Query Performance

| Query Type | Stages | Time | Notes |
|------------|--------|------|-------|
| Simple | Standard | ~900ms | Single retrieval + rerank |
| Comparison (2 entities) | Multihop | ~1600ms | 2 sub-queries + 3-stage rerank |
| Similarity (2 entities) | Multihop | ~1700ms | 2 sub-queries + fusion |
| Chaining (3 hops) | Multihop | ~2200ms | 3 sub-queries + global rerank |
| Aggregation | Multihop | ~2500ms | Multiple retrievals + fusion |

**Optimization Tips:**
- Use Ollama/vLLM for faster LLM inference
- Disable query rewriting for simple lookups
- Adjust `MULTIHOP_TOP_K_PER_SUBQUERY` for speed/quality tradeoff
- Use `use_reranker=false` in multihop endpoint for faster fusion-only

### Scalability

| Corpus Size | Index Time | Search Time | Memory |
|-------------|------------|-------------|--------|
| 10k docs    | ~5 min     | <10ms | 2GB    |
| 200k docs   | ~3 hours   | <15ms | 8GB    |
| 1M docs     | ~8 hours   | <30ms | 40GB   |

---

## 🎯 Roadmap

### ✅ Completed (v0.2)
- [x] Basic RAG pipeline (retrieval → LLM)
- [x] Cross-encoder reranking
- [x] Semantic chunking with LlamaIndex
- [x] Qdrant integration with HNSW
- [x] Progress tracking & resume
- [x] Multi-lingual support (PL/EN)
- [x] **Linguistic analysis with spaCy** 🆕
- [x] **Adaptive query rewriting** 🆕
- [x] **Multihop retrieval with three-stage reranking** 🆕
- [x] **Query type detection (8 types)** 🆕
- [x] **Multi-provider LLM support (HF/Ollama/vLLM/API)** 🆕
- [x] **Advanced prompting with templates** 🆕
- [x] **FastAPI REST server** 🆕

### 🚧 In Progress (v0.3)
- [ ] Self-Verification (CoVe) implementation
- [ ] Web UI for search
- [ ] Batch evaluation framework
- [ ] Query expansion with embedding similarity
- [ ] RAG fusion techniques

### 🔮 Planned (v0.4+)
- [ ] Hybrid search (BM25 + vector)
- [ ] Multi-modal support (images, tables)
- [ ] Multi-hop reasoning with graphs
- [ ] Fine-tuning scripts (LoRA/QLoRA)
- [ ] Deployment guides (Docker, K8s)
- [ ] Streaming responses
- [ ] Query caching

### Development Setup

```bash
# Install dev dependencies
make dev

# Pre-commit hooks
pre-commit install

# Code formatting
make fmt

# Linting
make lint
```

### Code Style
- **Formatter:** `black` + `isort`
- **Linter:** `ruff`
- **Type checking:** `mypy`
- **Docstrings:** Google style

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Papers & Methods
- **RAG:** [Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- **Query Rewriting:** [Query Rewriting for Retrieval (Jagerman et al., 2023)](https://arxiv.org/abs/2305.14283)
- **Multihop QA:** [HotpotQA (Yang et al., 2018)](https://arxiv.org/abs/1809.09600)
- **Cross-Encoders:** [Sentence-BERT (Reimers & Gurevych, 2019)](https://arxiv.org/abs/1908.10084)
- **CoVe:** [Chain-of-Verification (Dhuliawala et al., 2023)](https://arxiv.org/abs/2309.11495)

### Libraries & Tools
- [Sentence-Transformers](https://www.sbert.net/) - Embedding & reranking
- [Qdrant](https://qdrant.tech/) - Vector database
- [LlamaIndex](https://www.llamaindex.ai/) - Semantic chunking
- [spaCy](https://spacy.io/) - Linguistic analysis
- [Transformers](https://huggingface.co/transformers/) - LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) - REST API framework

### Models
- **GTE-multilingual** (Alibaba)
- **Jina-reranker-v2** (Jina AI)
- **Qwen3 / Qwen3** (Alibaba Cloud)


## 🔥 Quick Examples

### Example 1: Simple Lookup (Baseline)

```bash
$ curl -X POST "http://localhost:8000/ask/baseline" \
  -d '{"query": "Co to jest Warszawa?"}'

{
  "answer": "Warszawa to stolica Polski [1] i największe miasto kraju...",
  "metadata": {
    "pipeline": "baseline",
    "total_time_ms": 850.5
  }
}
```

### Example 2: Comparison Query (Enhanced with Multihop)

```bash
$ curl -X POST "http://localhost:8000/ask/enhanced" \
  -d '{"query": "ziemniaki vs pomidory błonnik"}'

{
  "answer": "Ziemniaki zawierają 2.2g błonnika na 100g [1][2], 
             pomidory 1.2g [3][4]. Ziemniaki mają więcej błonnika.",
  "metadata": {
    "is_multihop": true,
    "sub_queries": [
      "Ile błonnika mają ziemniaki?",
      "Ile błonnika mają pomidory?"
    ],
    "query_type": "comparison",
    "total_time_ms": 1576.6
  }
}
```

### Example 3: Verification Query

```bash
$ curl -X POST "http://localhost:8000/ask/enhanced" \
  -d '{"query": "Polska to największy kraj europejski?"}'

{
  "answer": "Nie, Polska nie jest największym krajem w Europie. 
             Polska ma powierzchnię 312,696 km² [1], podczas gdy 
             największym krajem Europy jest Rosja... [2]",
  "metadata": {
    "is_multihop": true,
    "query_type": "verification",
    "reasoning": "verification of superlative claim"
  }
}
```

### Example 4: Linguistic Analysis Only

```python
from src.ragx.retrieval.analyzers.linguistic_analyzer import LinguisticAnalyzer

analyzer = LinguisticAnalyzer()
features = analyzer.analyze("Co łączy mitologię słowiańską i nordycką?")

print(f"Tokens: {features.num_tokens}")
print(f"Clauses: {features.num_clauses}")
print(f"Entities: {features.entities}")
print(f"Has conjunctions: {features.has_conjunctions}")
```

### Example 5: Python API

```python
from src.ragx.pipelines.enhanced import EnhancedPipeline

# Initialize pipeline
pipeline = EnhancedPipeline()

# Ask a complex question
result = pipeline.answer(
    query="Król Mieszko I czy Bolesław Chrobry miał większy wpływ?",
    top_k=10
)

print(f"Answer: {result['answer']}")
print(f"Is multihop: {result['metadata']['is_multihop']}")
print(f"Sub-queries: {result['metadata']['sub_queries']}")
print(f"Query type: {result['metadata'].get('query_type')}")
print(f"Sources: {len(result['sources'])}")
```

---

**Happy RAG-ing! 🚀**
