# RAG Evaluation Framework

Comprehensive evaluation system for the RAG pipeline using RAGAS framework and ablation studies.

## Overview

This evaluation framework provides:

1. **Question Generation**: Generate test questions from Wikipedia articles
2. **RAGAS Evaluation**: Official RAGAS metrics + custom metrics
3. **Ablation Studies**: Compare different pipeline configurations
4. **Statistical Analysis**: T-tests and effect sizes

## Components

### 1. Question Generator (`generator/wikipedia_generator.py`)

Generates test questions from Wikipedia `.jsonl` files.

**Features**:
- Samples from Wikipedia article folders (AA-AJ by default)
- Generates 4 question types: simple, comparison, multihop, temporal
- Uses LLM to create realistic questions with ground truth
- Outputs RAGAS-compatible format

**Question Types**:
- **Simple** (40%): Factual questions from single article
- **Comparison** (25%): Compare/contrast two articles
- **Multihop** (25%): Requires reasoning across 2-3 articles
- **Temporal** (10%): Time-based or chronological questions

**Output Format**:
```json
{
  "question": "What is...?",
  "ground_truth": "Expected answer",
  "type": "simple|comparison|multihop|temporal",
  "source_urls": ["https://pl.wikipedia.org/wiki/..."],
  "contexts": ["Article text snippet..."]
}
```

### 2. RAGAS Evaluator (`ragas_evaluator.py`)

Evaluates RAG system using RAGAS framework + custom metrics.

**RAGAS Metrics** (official):
- **Faithfulness**: Factual accuracy of answer vs contexts
- **Answer Relevancy**: How relevant answer is to question
- **Context Precision**: Precision of retrieved contexts
- **Context Recall**: Recall of retrieved contexts vs ground truth

**Custom Metrics**:
- **Latency (ms)**: Total pipeline latency
- **Sources Count**: Number of unique sources retrieved
- **Multihop Coverage**: For multihop queries, % of sub-queries with ≥1 doc

### 3. Ablation Study (`ablation_study.py`)

Systematically test different pipeline configurations.

**Predefined Configurations**:
- `baseline`: No enhancements (vector search only)
- `query_only`: Query analysis/rewriting only
- `reranker_only`: Reranker only
- `cove_only`: CoVe verification only
- `no_cove`: Full pipeline without CoVe
- `full`: All enhancements enabled

**Features**:
- Batch evaluation on all test questions
- Statistical comparison (t-tests)
- Effect size calculation (Cohen's d)
- JSON export for reporting

## Usage

### Step 1: Generate Test Questions

```bash
# Generate 500 questions from Wikipedia articles
python scripts/generate_questions.py \
  --num-questions 500 \
  --data-dir data/processed/wiki_extracted \
  --output data/eval/generated_questions.jsonl

# Test with smaller sample (first 3 folders, 50 questions)
python scripts/generate_questions.py \
  --num-questions 50 \
  --folders AA AB AC \
  --output data/eval/test_questions.jsonl
```

**Requirements**:
- Wikipedia data in `data/processed/wiki_extracted/AA/`, `AB/`, etc.
- Each `.jsonl` file has: `{"id", "title", "text", "url"}`
- API LLM provider (OpenAI/Gemini) configured

### Step 2: Run Ablation Study

```bash
# Run full ablation study
python scripts/run_ablation_study.py \
  --questions data/eval/generated_questions.jsonl \
  --output results/ablation_study.json \
  --api-url http://localhost:8000

# Test with small sample
python scripts/run_ablation_study.py \
  --questions data/eval/test_questions.jsonl \
  --max-questions 20 \
  --output results/test_ablation.json

# Test specific configurations only
python scripts/run_ablation_study.py \
  --questions data/eval/generated_questions.jsonl \
  --configs baseline full no_cove \
  --output results/ablation_comparison.json
```

**Requirements**:
- RAG API running on `http://localhost:8000`
- `/eval/ablation` endpoint available
- OpenAI API key for RAGAS evaluation

### Step 3: Analyze Results

Results are saved to JSON with all metrics. The script also prints a summary table:

```
ABLATION STUDY RESULTS
================================================================================

Configuration        Faith     Rel    Prec  Recall  Latency  Sources Coverage
-------------------- ------- ------- ------- ------- --------- -------- --------
baseline               0.712   0.654   0.623   0.701     245ms      2.3    0.450
query_only             0.745   0.689   0.658   0.734     312ms      3.1    0.812
reranker_only          0.731   0.673   0.691   0.718     289ms      2.8    0.467
cove_only              0.798   0.702   0.641   0.723     567ms      2.4    0.455
no_cove                0.782   0.721   0.712   0.765     423ms      3.4    0.823
full                   0.823   0.738   0.729   0.781     612ms      3.6    0.831

BEST CONFIGURATIONS
================================================================================

Faithfulness        : full            (0.823)
Answer Relevancy    : full            (0.738)
Context Precision   : full            (0.729)
Context Recall      : full            (0.781)

STATISTICAL COMPARISONS (t-tests)
================================================================================

Full vs Baseline (Faithfulness):
  Full:     0.823
  Baseline: 0.712
  Diff:     +0.111
  p-value:  0.0023 ✓ SIGNIFICANT
  Effect:   large (d=1.23)
```

## Python API Usage

### Programmatic Evaluation

```python
from pathlib import Path
from src.ragx.evaluation import RAGASEvaluator, AblationStudy

# Initialize evaluator
evaluator = RAGASEvaluator(llm_model="gpt-4o-mini")

# Single question evaluation
result = evaluator.evaluate_single(
    question="What is the capital of Poland?",
    answer="The capital of Poland is Warsaw.",
    contexts=["Warsaw is the capital and largest city of Poland..."],
    ground_truth="Warsaw",
    metadata={"latency_ms": 245, "sources": ["url1"]},
)

print(f"Faithfulness: {result.faithfulness:.3f}")
print(f"Answer Relevancy: {result.answer_relevancy:.3f}")

# Batch evaluation
results = evaluator.evaluate_batch(
    questions=[...],
    answers=[...],
    contexts_list=[...],
    ground_truths=[...],
    metadata_list=[...],
)

print(f"Mean Faithfulness: {results.mean_faithfulness:.3f}")
```

### Programmatic Ablation Study

```python
from src.ragx.evaluation import AblationStudy, PipelineConfig

# Initialize
ablation = AblationStudy(
    api_base_url="http://localhost:8000",
)

# Custom configuration
custom_config = PipelineConfig(
    name="custom",
    description="Query analysis + CoVe only",
    query_analysis_enabled=True,
    reranker_enabled=False,
    cove_enabled=True,
    multihop_enabled=True,
)

# Run study
result = ablation.run(
    questions_path=Path("data/eval/questions.jsonl"),
    configs=[AblationStudy.BASELINE, custom_config, AblationStudy.FULL],
)

# Get best config
best = result.get_best_config("mean_faithfulness")
print(f"Best: {best.config.name} ({best.evaluation.mean_faithfulness:.3f})")

# Compare configs
comparison = result.compare_configs("full", "baseline", "mean_faithfulness")
print(f"Difference: {comparison['mean_diff']:+.3f}")
print(f"p-value: {comparison['p_value']:.4f}")
print(f"Effect: {comparison['effect_size']}")
```

## Metrics Explained

### RAGAS Metrics

**Faithfulness** (0.0-1.0):
- Measures factual accuracy of answer against retrieved contexts
- Higher = answer is more grounded in evidence
- LLM-based: extracts claims, checks if supported by contexts

**Answer Relevancy** (0.0-1.0):
- Measures how well answer addresses the question
- Higher = more relevant/on-topic answer
- LLM-based: compares question and answer semantically

**Context Precision** (0.0-1.0):
- Precision of retrieved contexts (are they all relevant?)
- Higher = less noise in retrieved contexts
- Ground truth-based: checks if contexts help answer question

**Context Recall** (0.0-1.0):
- Recall of retrieved contexts (did we get everything needed?)
- Higher = better coverage of necessary information
- Ground truth-based: checks if all info from ground truth is present

### Custom Metrics

**Latency (ms)**:
- Total pipeline execution time
- Includes: query rewriting, retrieval, reranking, generation, CoVe
- Lower = faster pipeline

**Sources Count**:
- Number of unique sources (URLs, doc IDs) retrieved
- For multihop: shows diversity of sources
- Higher = more diverse information

**Multihop Coverage** (0.0-1.0):
- For multihop queries: ratio of sub-queries with ≥1 retrieved doc
- 1.0 = every sub-query got at least one document
- 0.5 = only half the sub-queries got documents
- N/A for single-hop queries (defaults to 1.0)

## Integration with Pipeline

The ablation study uses the existing `/eval/ablation` endpoint which supports toggling:

```json
{
  "query": "What is...?",
  "config": {
    "query_analysis_enabled": true,
    "reranker_enabled": true,
    "cove_enabled": true,
    "multihop_enabled": true
  },
  "top_k": 5
}
```

This endpoint returns:
```json
{
  "answer": "Generated answer...",
  "contexts": ["context1", "context2"],
  "sub_queries": ["sub1", "sub2"],
  "metadata": {
    "total_time_ms": 612,
    "is_multihop": true,
    "query_type": "comparison"
  },
  "context_details": [
    {"url": "https://...", "score": 0.89}
  ]
}
```

## Dependencies

Required packages (add to `requirements.txt`):

```
ragas>=0.1.0
datasets>=2.14.0
langchain-openai>=0.0.2
scipy>=1.11.0
```

## File Structure

```
src/ragx/evaluation/
├── __init__.py                      # Main exports
├── README.md                        # This file
├── ragas_evaluator.py              # RAGAS + custom metrics
├── ablation_study.py               # Ablation study runner
└── generator/
    ├── __init__.py
    └── wikipedia_generator.py      # Question generation

scripts/
├── generate_questions.py           # CLI for question generation
└── run_ablation_study.py          # CLI for ablation study

data/
└── eval/
    ├── generated_questions.jsonl   # Generated test questions
    └── test_questions.jsonl        # Small test sample

results/
└── ablation_study.json            # Evaluation results
```

## Next Steps

1. **Generate Questions**: Populate `data/processed/wiki_extracted/` with Wikipedia data, then run question generator
2. **Test Pipeline**: Ensure RAG API is running with `/eval/ablation` endpoint
3. **Small Test**: Run ablation study on 20 questions to verify setup
4. **Full Evaluation**: Run on 500 questions for comprehensive analysis
5. **Reporting**: Create HTML reports with visualizations (future work)

## Tips

- Start with `--max-questions 20` to test before full run
- Use `gpt-4o-mini` for RAGAS (cheaper than gpt-4)
- Multihop coverage metric is key for evaluating multihop improvements
- CoVe adds ~300ms latency but improves faithfulness significantly
- Check p-values < 0.05 for significant differences
