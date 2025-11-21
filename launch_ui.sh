#!/bin/bash
# Launch RAGx Interactive Chat UI

echo "🚀 Starting RAGx Chat UI..."
echo ""
echo "Make sure the API server is running:"
echo "  python -m src.ragx.api.main"
echo ""
echo "Opening UI at http://localhost:8501"
echo ""

streamlit run src/ragx/ui/chat_app.py
