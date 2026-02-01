# RAGx Interactive Chat UI

Interactive chat interface for RAGx with real-time pipeline status visualization.

## Features

- **Live Status Display** - See each pipeline step as it happens
- **Configuration Presets** - Quick access to baseline, enhanced, and custom configs
- **Timing Visualization** - Interactive charts for pipeline phase breakdown
- **Source Citations** - Expandable source details with scores and Wikipedia links
- **A/B Comparison Mode** - Compare baseline vs enhanced side-by-side
- **Example Queries** - Pre-built questions for different scenarios
- **Session Statistics** - Track queries, timing, and config usage
- **Export Sessions** - Save chat as JSON or Markdown
- **Modern UI** - Clean, responsive Streamlit interface

## Quick Start

### 1. Install Dependencies
```
bash
pip install streamlit requests
```
### 2. Start RAGx API Server
```
bash
# Terminal 1
python -m src.ragx.api.main
```
The API should be running on `http://localhost:8000`.

### 3. Launch Chat UI
```
bash
# Terminal 2
streamlit run src/ragx/ui/chat_app.py
```

The UI will open at `http://localhost:8501`.

## Usage

### Configuration Modes

#### Presets
Choose from predefined configurations:
- **Baseline** - Vector search only (no enhancements)
- **Enhanced (Full)** - All features except CoVe
- **Enhanced + CoVe** - Full pipeline with verification
- **Query Analysis Only** - Multihop detection only
- **Reranker Only** - Semantic reranking only

#### Custom
Fine-tune individual components:
- **Query Analysis** - Multihop detection and adaptive rewriting
- **Chain of Thought** - Step-by-step reasoning
- **Reranker** - Semantic reranking (3-stage for multihop)
- **CoVe Mode** - Verification (off/auto/metadata/suggest)
- **Prompt Template** - auto/basic/enhanced
- **Top K** - Number of contexts (1-20)

### Live Pipeline Status

Watch your query flow through the pipeline:
```
text
Step 1/5: Analyzing query...
Step 2/5: Retrieving candidates...
Step 3/5: Reranking results...
Step 4/5: Generating answer...
Step 5/5: Verifying with CoVe...
Complete. Total: 1234ms
```
### Detailed Metrics

Expand the timing and pipeline info sections to see:
- **Timing breakdown** - ms spent in each phase and interactive chart
- **Query analysis** - Detected type, sub-queries, reasoning
- **Retrieval stats** - Candidates retrieved, final sources
- **Template used** - Selected prompt template
- **Source details** - Full text, scores, Wikipedia URLs

### A/B Comparison Mode

Enable comparison mode in sidebar to test the same query with different configs:

1. Enable "A/B Comparison Mode"
2. Enter your question
3. View baseline vs enhanced results side-by-side
4. Compare timing, sources, and answer quality

Useful for ablation-style evaluation.

### Example Queries

Use sidebar examples to test quickly:
- **Simple** - Basic factual questions
- **Multihop** - Comparison and multi-part questions
- **Complex** - Advanced reasoning queries

### Session Statistics and Export

**View Stats:**
- Click "View Session Stats" to see:
  - Total queries processed
  - Average response time
  - Config usage breakdown

**Export Session:**
- Click "Export Session"
- Choose format:
  - **JSON** - Full metadata, sources, timings
  - **Markdown** - Clean readable format
- Download a timestamped file

## Architecture
```mermaid 
User Input
    |
    v
[Streamlit UI] -> API Request
    |
    v
[FastAPI Server] -> /eval/ablation endpoint
    |
    v
Pipeline Steps (configurable):
    1. Query Analysis (optional)
    2. Retrieval
    3. Reranking (optional)
    4. Generation
    5. CoVe Verification (optional)
    |
    v
[Response] -> UI Display
```
## Development

### Project Structure
```
text
src/ragx/ui/
├── __init__.py
├── chat_app.py              # Main entry point
├── types.py                 # Type definitions
├── helpers.py               # Helper functions
├── config/
│   ├── __init__.py
│   ├── presets.py          # Pipeline configurations
│   └── session_state.py    # State initialization
├── components/
│   ├── __init__.py
│   ├── sidebar.py          # Config and features
│   ├── chat_display.py     # Message rendering
│   └── progress.py         # Progress tracking
└── README.md               # This file
```
### Code Architecture

**Core Types** (`types.py`):
- `PipelineConfig` - Configuration dataclass
- `PipelineStep` - Step with message and timing
- `StepTiming` - Duration estimates per phase

**Helpers** (`helpers.py`):
- `estimate_step_timings()` - Timing estimation
- `get_pipeline_steps()` - Step list with numbering
- `call_rag_api()` - API communication
- `update_session_stats()` - Statistics tracking

**Configuration** (`config/`):
- `presets.py` - Predefined pipeline configs
- `session_state.py` - Centralized session state initialization

**UI Components** (`components/`):
- `sidebar.py` - Configuration panel, features, stats, export
- `chat_display.py` - Message history, metadata, sources, charts
- `progress.py` - Threading-based progress tracking

**Main App** (`chat_app.py`):
- Page configuration and layout
- Chat loop and message handling
- Query processing with progress display
- A/B comparison mode logic

### Timing Estimates

Based on typical enhanced pipeline performance (~30-32 seconds):

| Phase | Duration | Notes |
|-------|----------|-------|
| Query Analysis | 1.5s | Multihop detection and rewriting |
| Retrieval | 6.0s | Vector search and embeddings |
| Reranking | 4.0s | 3-stage for multihop queries |
| Generation | 14.0s (CoT) / 8.0s | LLM inference |
| CoVe | 6.5s | Verification queries |

Total: ~32s for full enhanced pipeline.

### Extending the UI

**Add New Presets:**
1. Edit `PRESETS` in `config/presets.py`
2. Add a new `PipelineConfig`
3. The preset appears in the dropdown automatically

**Add New Components:**
1. Create a module in `components/`
2. Import it in `components/__init__.py`
3. Call it from `chat_app.py`

**Modify Timing Estimates:**
1. Update `estimate_step_timings()` in `helpers.py`
2. Adjust `StepTiming` values based on profiling

**Custom Visualizations:**
1. Extend `_render_message_metadata()` in `chat_display.py`
2. Use Plotly (with fallback)
3. Keep charts in expanders to avoid clutter

**Styling:**
- Configure Streamlit theme in `.streamlit/config.toml`
- Use consistent visual hierarchy for labels and sections
- Customize Plotly theme in chart creation

## Troubleshooting

### API Connection Failed
- Ensure the API server is running: `python -m src.ragx.api.main`
- Check API URL in sidebar (default: `http://localhost:8000`)
- Verify: `curl http://localhost:8000/health`

### Request Timeout
- Increase timeout in `chat_app.py` (default: 120s)
- Check API logs for errors
- Try simpler queries with baseline config

### Slow Performance
- Reduce `top_k`
- Disable expensive features (CoVe, CoT)
- Verify CPU/GPU settings in `.env`

## Advanced Usage

### Run on a Different Port
```
bash
streamlit run src/ragx/ui/chat_app.py --server.port 8502
```
### Custom API URL

Set in the sidebar or via environment:
```
bash
export RAGX_API_URL="http://your-api:8000"
streamlit run src/ragx/ui/chat_app.py
```
---
