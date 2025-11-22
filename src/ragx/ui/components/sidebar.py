import streamlit as st
import requests
import json
from datetime import datetime

from src.ragx.ui.constants.types import PipelineConfig
from ..config import PRESETS

def render_sidebar(api_url: str) -> PipelineConfig:
    """
    Render sidebar with configuration options.

    Returns:
        Selected PipelineConfig
    """
    st.title("⚙️ Configuration")

    # API URL configuration
    api_url = st.text_input(
        "API URL",
        value=api_url,
        help="Base URL of the RAGx API server"
    )
    st.session_state.api_url = api_url

    st.divider()

    # Configuration mode selection
    config_mode = st.radio(
        "Configuration Mode",
        options=["Presets", "Custom"],
        help="Use predefined presets or configure manually"
    )

    if config_mode == "Presets":
        preset_key = st.selectbox(
            "Select Pipeline",
            options=list(PRESETS.keys()),
            format_func=lambda x: PRESETS[x].name,
            help="Choose a predefined configuration"
        )

        preset = PRESETS[preset_key]
        st.info(preset.description)

        # Display current config
        with st.expander("📋 Configuration Details"):
            st.json({
                "query_analysis": preset.query_analysis_enabled,
                "cot": preset.cot_enabled,
                "reranker": preset.reranker_enabled,
                "cove_mode": preset.cove_mode,
                "prompt_template": preset.prompt_template,
                "top_k": preset.top_k,
            })

    else:  # Custom mode
        st.markdown("### Pipeline Toggles")

        query_analysis = st.checkbox(
            "🔍 Query Analysis",
            value=True,
            help="Enable multihop detection and adaptive rewriting"
        )

        cot = st.checkbox(
            "🧠 Chain of Thought",
            value=True,
            help="Enable step-by-step reasoning"
        )

        reranker = st.checkbox(
            "📊 Reranker",
            value=True,
            help="Enable semantic reranking"
        )

        cove_mode = st.selectbox(
            "✅ CoVe Mode",
            options=["off", "auto", "metadata", "suggest"],
            help="Chain-of-Verification mode"
        )

        prompt_template = st.selectbox(
            "📝 Prompt Template",
            options=["auto", "basic", "enhanced"],
            help="Auto adapts based on query analysis"
        )

        top_k = st.slider(
            "🔢 Top K",
            min_value=1,
            max_value=20,
            value=10,
            help="Number of contexts to retrieve"
        )

        preset = PipelineConfig(
            name="Custom",
            description="Custom configuration",
            query_analysis_enabled=query_analysis,
            cot_enabled=cot,
            reranker_enabled=reranker,
            cove_mode=cove_mode,
            prompt_template=prompt_template,
            top_k=top_k,
        )

    st.divider()

    # Advanced features
    _render_advanced_features()

    st.divider()

    # Example queries
    _render_example_queries()

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_stats = {
            "total_queries": 0,
            "total_time_ms": 0,
            "configs_used": {},
        }
        st.rerun()

    st.divider()

    # Connection status
    _render_connection_status(api_url)

    return preset


def _render_advanced_features():
    """Render advanced features section."""
    st.markdown("### 🔬 Advanced Features")

    # A/B Comparison Mode
    st.session_state.comparison_mode = st.checkbox(
        "🔀 A/B Comparison Mode",
        value=st.session_state.comparison_mode,
        help="Send query to 2 configs and compare side-by-side"
    )

    if st.session_state.comparison_mode:
        st.info("💡 Next query will be sent to both Baseline and Enhanced configs")

    # Session Statistics
    if st.button("📊 View Session Stats"):
        stats = st.session_state.session_stats
        if stats["total_queries"] > 0:
            st.metric("Total Queries", stats["total_queries"])
            st.metric("Avg Time", f"{stats['total_time_ms'] / stats['total_queries']:.0f}ms")

            if stats["configs_used"]:
                st.write("**Configs Used:**")
                for cfg, count in stats["configs_used"].items():
                    st.write(f"- {cfg}: {count}x")
        else:
            st.info("No queries yet")

    # Export Session
    if st.button("💾 Export Session"):
        if st.session_state.messages:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "stats": st.session_state.session_stats,
                "messages": st.session_state.messages,
            }

            # JSON download
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"ragx_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

            # Markdown download
            md_content = f"# RAGx Chat Session\n\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            for msg in st.session_state.messages:
                role = "**User:**" if msg["role"] == "user" else "**Assistant:**"
                md_content += f"{role}\n{msg['content']}\n\n"
                if msg["role"] == "assistant" and "sources" in msg:
                    md_content += f"*Sources: {len(msg['sources'])} documents*\n\n"

            st.download_button(
                label="📥 Download Markdown",
                data=md_content,
                file_name=f"ragx_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
        else:
            st.info("No messages to export")


def _render_example_queries():
    """Render example queries section."""
    st.markdown("### 💡 Example Queries")

    example_queries = {
        "🔵 Simple": [
            "Czym jest sztuczna inteligencja?",
            "Kiedy powstała Wikipedia?",
        ],
        "🟣 Multihop": [
            "Porównaj mitologię słowiańską i nordycką",
            "Ziemniaki vs pomidory - który ma więcej błonnika?",
            "Jakie są podobieństwa między kwantową mechaniką a teorią względności?",
        ],
        "🟢 Complex": [
            "Jak rozwój AI wpływa na rynek pracy i które zawody są najbardziej zagrożone?",
        ],
    }

    for category, queries in example_queries.items():
        with st.expander(category):
            for q in queries:
                if st.button(q, key=f"example_{hash(q)}"):
                    st.session_state.example_query = q
                    st.rerun()


def _render_connection_status(api_url: str):
    """Render connection status checker."""
    st.markdown("### Connection Status")

    # Only check on button click (avoid spam)
    if st.button("🔄 Check connection"):
        try:
            response = requests.get(f"{api_url}/info/health", timeout=2)
            if response.ok:
                st.session_state.connection_status = {
                    "status": "ok",
                    "message": "Connected",
                }
            else:
                st.session_state.connection_status = {
                    "status": "error",
                    "message": f"API Error (status {response.status_code})",
                }
        except Exception:
            st.session_state.connection_status = {
                "status": "error",
                "message": "Not Connected",
            }

    # Display cached status
    status = st.session_state.connection_status
    if status["status"] == "ok":
        st.success(f"✅ {status['message']}")
    elif status["status"] == "error":
        st.error(f"❌ {status['message']}")
    else:
        st.info(status["message"])
