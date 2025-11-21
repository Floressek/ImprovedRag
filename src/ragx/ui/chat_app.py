"""
RAGx Interactive Chat UI with Live Status Display

Features:
- Real-time pipeline step visualization
- Pipeline configuration selector (baseline/enhanced/custom)
- Detailed timing and metadata display
- Source citations with expandable details
"""

from __future__ import annotations

import streamlit as st
import requests
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

# Page config
st.set_page_config(
    page_title="RAGx Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CONFIGURATION PRESETS (from ablation_study.py)
# ============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration preset."""
    name: str
    description: str
    query_analysis_enabled: bool = True
    cot_enabled: bool = True
    reranker_enabled: bool = True
    cove_mode: str = "off"
    prompt_template: str = "auto"
    top_k: int = 10

PRESETS = {
    "baseline": PipelineConfig(
        name="Baseline",
        description="🔵 No enhancements (vector search only)",
        query_analysis_enabled=False,
        cot_enabled=False,
        reranker_enabled=False,
        cove_mode="off",
        prompt_template="basic",
    ),
    "enhanced_full": PipelineConfig(
        name="Enhanced (Full)",
        description="🟢 All enhancements enabled (no CoVe)",
        query_analysis_enabled=True,
        cot_enabled=True,
        reranker_enabled=True,
        cove_mode="off",
        prompt_template="auto",
    ),
    "enhanced_cove": PipelineConfig(
        name="Enhanced + CoVe",
        description="🟣 Full pipeline with CoVe auto-correction",
        query_analysis_enabled=True,
        cot_enabled=True,
        reranker_enabled=True,
        cove_mode="auto",
        prompt_template="auto",
    ),
    "query_only": PipelineConfig(
        name="Query Analysis Only",
        description="🟡 Multihop detection only",
        query_analysis_enabled=True,
        cot_enabled=False,
        reranker_enabled=False,
        cove_mode="off",
        prompt_template="basic",
    ),
    "reranker_only": PipelineConfig(
        name="Reranker Only",
        description="🟠 Reranking only",
        query_analysis_enabled=False,
        cot_enabled=False,
        reranker_enabled=True,
        cove_mode="off",
        prompt_template="basic",
    ),
}

# ============================================================================
# SESSION STATE
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")

    # API URL
    api_url = st.text_input(
        "API URL",
        value=st.session_state.api_url,
        help="Base URL of the RAGx API server"
    )
    st.session_state.api_url = api_url

    st.divider()

    # Configuration Mode
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

        # Create custom preset
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

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Connection status
    st.markdown("### Connection Status")
    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        if response.ok:
            st.success("✅ Connected")
        else:
            st.error("❌ API Error")
    except Exception as e:
        st.error(f"❌ Not Connected")

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

st.title("💬 RAGx Interactive Chat")
st.caption(f"Using pipeline: **{preset.name}**")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show metadata for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]

            # Timing summary
            with st.expander("⏱️ Timing Details"):
                timings = metadata.get("timings", {})
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total", f"{timings.get('total_time_ms', 0):.0f}ms")
                    st.metric("Retrieval", f"{timings.get('retrieval_time_ms', 0):.0f}ms")

                with col2:
                    st.metric("Reranking", f"{timings.get('rerank_time_ms', 0):.0f}ms")
                    st.metric("Generation", f"{timings.get('llm_time_ms', 0):.0f}ms")

                with col3:
                    st.metric("Query Analysis", f"{timings.get('rewrite_time_ms', 0):.0f}ms")
                    if timings.get('cove_time_ms', 0) > 0:
                        st.metric("CoVe", f"{timings.get('cove_time_ms', 0):.0f}ms")

            # Pipeline info
            with st.expander("🔧 Pipeline Info"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Template Used:** {metadata.get('template_used', 'N/A')}")
                    st.write(f"**Query Type:** {metadata.get('query_type', 'N/A')}")
                    st.write(f"**Multihop:** {'Yes' if metadata.get('is_multihop') else 'No'}")

                with col2:
                    st.write(f"**Candidates:** {metadata.get('num_candidates', 0)}")
                    st.write(f"**Sources Used:** {metadata.get('num_sources', 0)}")
                    phases = [p for p in metadata.get('phases', []) if p]
                    st.write(f"**Phases:** {len(phases)}")

                # Sub-queries if multihop
                if metadata.get('is_multihop') and metadata.get('sub_queries'):
                    st.markdown("**Sub-queries:**")
                    for i, sq in enumerate(metadata.get('sub_queries', []), 1):
                        st.write(f"{i}. {sq}")

            # Sources
            if "sources" in message and message["sources"]:
                with st.expander(f"📚 Sources ({len(message['sources'])})", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        # Source header with title
                        st.markdown(f"### [{i}] {source.get('doc_title', 'Unknown')}")

                        # Metadata badges
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if source.get('position') is not None:
                                st.caption(f"📄 Chunk {source.get('position', 0) + 1}/{source.get('total_chunks', '?')}")
                        with col2:
                            if source.get('retrieval_score') is not None:
                                st.caption(f"🎯 Score: {source.get('retrieval_score', 0):.3f}")
                        with col3:
                            if source.get('rerank_score') is not None:
                                st.caption(f"📊 Rerank: {source.get('rerank_score', 0):.3f}")

                        # Full text with expandable view
                        source_text = source.get('text', '')
                        if len(source_text) > 300:
                            with st.expander("📖 View full text"):
                                st.text_area(
                                    "Source text",
                                    source_text,
                                    height=200,
                                    disabled=True,
                                    label_visibility="collapsed",
                                    key=f"source_{i}_{message.get('timestamp', '')}"
                                )
                                # Copy button
                                if st.button("📋 Copy text", key=f"copy_{i}_{message.get('timestamp', '')}"):
                                    st.code(source_text, language=None)
                                    st.success("✓ Text displayed above - use browser copy")
                            st.caption(source_text[:300] + "...")
                        else:
                            st.info(source_text)

                        # Clickable URL link
                        if source.get('url'):
                            st.markdown(f"🔗 [Open Wikipedia article]({source['url']})")

                        # Additional metadata for multihop
                        if source.get('source_subquery'):
                            st.caption(f"💡 From sub-query: _{source.get('source_subquery')}_")

                        # CoVe evidence marker
                        if source.get('source') == "EVIDENCE FROM COVE":
                            st.success("✅ Additional evidence from CoVe verification")

                        st.divider()

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": time.time()
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with RAG pipeline
    with st.chat_message("assistant"):
        # Live status display
        status_container = st.status("🔄 Processing query...", expanded=True)

        with status_container:
            try:
                # Prepare request
                request_data = {
                    "query": prompt,
                    "use_query_analysis": preset.query_analysis_enabled,
                    "use_cot": preset.cot_enabled,
                    "use_reranker": preset.reranker_enabled,
                    "cove": preset.cove_mode,
                    "prompt_template": preset.prompt_template,
                    "top_k": preset.top_k,
                }

                # Track timing
                start_time = time.time()

                # Step 1: Query Analysis
                if preset.query_analysis_enabled:
                    st.write("🔍 **Step 1/5:** Analyzing query...")
                else:
                    st.write("🔍 **Step 1/5:** Skipped (disabled)")

                # Step 2: Retrieval
                st.write("📥 **Step 2/5:** Retrieving candidates...")

                # Make API request
                response = requests.post(
                    f"{api_url}/eval/ablation",
                    json=request_data,
                    timeout=120,
                )

                response.raise_for_status()
                result = response.json()

                # Step 3: Reranking
                if preset.reranker_enabled:
                    st.write("📊 **Step 3/5:** Reranking results...")
                else:
                    st.write("📊 **Step 3/5:** Skipped (disabled)")

                # Step 4: Generation
                st.write("💭 **Step 4/5:** Generating answer...")

                # Step 5: CoVe
                if preset.cove_mode != "off":
                    st.write("✅ **Step 5/5:** Verifying with CoVe...")
                else:
                    st.write("✅ **Step 5/5:** Skipped (disabled)")

                total_time = (time.time() - start_time) * 1000
                st.write(f"✨ **Complete!** Total: {total_time:.0f}ms")

            except requests.exceptions.RequestException as e:
                st.error(f"❌ API Error: {str(e)}")
                status_container.update(label="❌ Request failed", state="error")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                status_container.update(label="❌ Processing failed", state="error")
                st.stop()

        # Update status to complete
        status_container.update(label="✅ Processing complete", state="complete", expanded=False)

        # Display answer
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        metadata = result.get("metadata", {})

        st.markdown(answer)

        # Add to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "metadata": metadata,
            "timestamp": time.time()
        })

        st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.caption("RAGx Chat UI v1.0")
st.sidebar.caption("Built with Streamlit")
