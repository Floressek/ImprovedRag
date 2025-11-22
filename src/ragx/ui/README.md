# RAGx Interactive Chat UI 💬

Beautiful, interactive chat interface for RAGx with **real-time pipeline status visualization**.

## Features

- 🎯 **Live Status Display** - See each pipeline step as it happens
- ⚙️ **Configuration Presets** - Quick access to baseline, enhanced, and custom configs
- 📊 **Timing Visualization** - Interactive charts for pipeline phase breakdown
- 🔍 **Source Citations** - Expandable source details with scores and Wikipedia links
- 🔀 **A/B Comparison Mode** - Compare baseline vs enhanced side-by-side
- 💡 **Example Queries** - Pre-built questions for different scenarios
- 📈 **Session Statistics** - Track queries, timing, and config usage
- 💾 **Export Sessions** - Save chat as JSON or Markdown
- 🎨 **Modern UI** - Clean, responsive Streamlit interface

## Quick Start

### 1. Install Dependencies

```bash
pip install streamlit requests
```

### 2. Start RAGx API Server

```bash
# In terminal 1
python -m src.ragx.api.main
```

The API should be running on `http://localhost:8000`

### 3. Launch Chat UI

```bash
# In terminal 2
streamlit run src/ragx/ui/chat_app.py
```

The UI will open at `http://localhost:8501`

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
- 🔍 **Query Analysis** - Multihop detection & adaptive rewriting
- 🧠 **Chain of Thought** - Step-by-step reasoning
- 📊 **Reranker** - Semantic reranking (3-stage for multihop)
- ✅ **CoVe Mode** - Verification (off/auto/metadata/suggest)
- 📝 **Prompt Template** - auto/basic/enhanced
- 🔢 **Top K** - Number of contexts (1-20)

### Live Pipeline Status

Watch your query flow through the pipeline:

```
🔍 Step 1/5: Analyzing query...
📥 Step 2/5: Retrieving candidates...
📊 Step 3/5: Reranking results...
💭 Step 4/5: Generating answer...
✅ Step 5/5: Verifying with CoVe...
✨ Complete! Total: 1234ms
```

### Detailed Metrics

Expand the timing and pipeline info sections to see:
- **Timing breakdown** - ms spent in each phase + interactive chart
- **Query analysis** - Detected type, sub-queries, reasoning
- **Retrieval stats** - Candidates retrieved, final sources
- **Template used** - Which prompt template was selected
- **Source details** - Full text, scores, Wikipedia URLs

### A/B Comparison Mode

Enable comparison mode in sidebar to test the same query with different configs:

1. ✅ Enable "🔀 A/B Comparison Mode" checkbox
2. Enter your question
3. See baseline vs enhanced results side-by-side
4. Compare timing, sources, and answer quality

Perfect for ablation study analysis!

### Example Queries

Click example queries in sidebar to quick-test:
- **🔵 Simple** - Basic factual questions
- **🟣 Multihop** - Comparison and multi-part questions
- **🟢 Complex** - Advanced reasoning queries

### Session Statistics & Export

**View Stats:**
- Click "📊 View Session Stats" to see:
  - Total queries processed
  - Average response time
  - Config usage breakdown

**Export Session:**
- Click "💾 Export Session"
- Choose format:
  - **JSON** - Full metadata, sources, timings
  - **Markdown** - Clean readable format
- Download timestamped file

## Architecture

```
User Input
    ↓
[Streamlit UI] → API Request
    ↓
[FastAPI Server] → /eval/ablation endpoint
    ↓
Pipeline Steps (with toggles):
    1. Query Analysis (optional)
    2. Retrieval
    3. Reranking (optional)
    4. Generation
    5. CoVe Verification (optional)
    ↓
[Response] → UI Display
```

## Development

### Project Structure

Modular architecture with clean separation of concerns:

```
src/ragx/ui/
├── __init__.py
├── chat_app.py              # Main entry point (202 lines)
├── types.py                 # Type definitions (47 lines)
├── helpers.py               # Helper functions (145 lines)
├── config/
│   ├── __init__.py
│   ├── presets.py          # Pipeline configurations
│   └── session_state.py    # State initialization
├── components/
│   ├── __init__.py
│   ├── sidebar.py          # Config & features (264 lines)
│   ├── chat_display.py     # Message rendering (158 lines)
│   └── progress.py         # Progress tracking (115 lines)
└── README.md               # This file
```

### Code Architecture

**Core Types** (`types.py`):
- `PipelineConfig` - Configuration dataclass
- `PipelineStep` - Step with message and timing
- `StepTiming` - Duration estimates per phase

**Helpers** (`helpers.py`):
- `estimate_step_timings()` - Realistic timing calculation
- `get_pipeline_steps()` - Step list with numbering
- `call_rag_api()` - API communication
- `update_session_stats()` - Statistics tracking

**Configuration** (`config/`):
- `presets.py` - Predefined pipeline configs (baseline, enhanced, etc.)
- `session_state.py` - Centralized session state initialization

**UI Components** (`components/`):
- `sidebar.py` - Configuration panel, features, stats, export
- `chat_display.py` - Message history, metadata, sources, charts
- `progress.py` - Threading-based real-time progress tracking

**Main App** (`chat_app.py`):
- Page configuration and layout
- Chat loop and message handling
- Query processing with progress display
- A/B comparison mode logic

### Realistic Timing Estimates

Based on actual Enhanced pipeline performance (~30-32 seconds):

| Phase | Duration | Notes |
|-------|----------|-------|
| Query Analysis | 1.5s | Multihop detection + rewriting |
| Retrieval | 6.0s | Vector search + embeddings |
| Reranking | 4.0s | 3-stage for multihop queries |
| Generation | 14.0s (CoT) / 8.0s | LLM inference (longest step) |
| CoVe | 6.5s | Verification queries |

Total: ~32s for full Enhanced pipeline

### Extending the UI

**Add New Presets:**
1. Edit `PRESETS` dict in `config/presets.py`
2. Create new `PipelineConfig` instance
3. Available in dropdown automatically

**Add New Components:**
1. Create module in `components/`
2. Import in `components/__init__.py`
3. Call from `chat_app.py` main loop

**Modify Timing Estimates:**
1. Edit `estimate_step_timings()` in `helpers.py`
2. Adjust `StepTiming` values based on profiling

**Custom Visualizations:**
1. Add to `chat_display.py` in `_render_message_metadata()`
2. Use Plotly (with fallback) for charts
3. Wrap in expanders to keep UI clean

**Styling:**
- Use Streamlit theming in `.streamlit/config.toml`
- Consistent emoji prefixes for visual hierarchy
- Plotly theme customization in chart creation

## Troubleshooting

### API Connection Failed
- Ensure API server is running: `python -m src.ragx.api.main`
- Check API URL in sidebar (default: `http://localhost:8000`)
- Verify with: `curl http://localhost:8000/health`

### Request Timeout
- Increase timeout in `chat_app.py` (default: 120s)
- Check API logs for errors
- Try simpler queries with baseline config

### Slow Performance
- Reduce `top_k` value
- Disable expensive features (CoVe, CoT)
- Use CPU/GPU settings in `.env`

## Advanced Usage

### Running on Different Port

```bash
streamlit run src/ragx/ui/chat_app.py --server.port 8502
```

### Custom API URL

Set in sidebar or via environment:

```bash
export RAGX_API_URL="http://your-api:8000"
streamlit run src/ragx/ui/chat_app.py
```

## Screenshots

### Configuration Panel
Sidebar with presets and custom toggles

### Live Status
Real-time pipeline execution steps

### Results Display
Answer with expandable timing, sources, and metadata

---

**Built with ❤️ using Streamlit**
