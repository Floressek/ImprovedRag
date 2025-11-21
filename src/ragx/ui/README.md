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

```
src/ragx/ui/
├── __init__.py
├── chat_app.py       # Main Streamlit app
└── README.md         # This file
```

### Extending the UI

To add new features:

1. **New Presets**: Edit `PRESETS` dict in `chat_app.py`
2. **Custom Metrics**: Add to the metadata expanders
3. **Styling**: Use Streamlit theming in `.streamlit/config.toml`

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
