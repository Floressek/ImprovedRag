import streamlit as st
import time
import json
from datetime import datetime

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))


from src.ragx.ui.config import PRESETS, initialize_session_state
from src.ragx.ui.components import (
    render_sidebar,
    render_message_history,
    show_progress_with_api_call,
    _render_message_metadata,
    _render_sources,
)
from src.ragx.ui.helpers.helpers import update_session_stats

# Page configuration
st.set_page_config(
    page_title="RAGx Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session_state()


def _reset_chat():
    """Reset chat messages and session statistics."""
    st.session_state.messages = []
    st.session_state.session_stats = {
        "total_queries": 0,
        "total_time_ms": 0,
        "configs_used": {},
    }


def _run_single_query(prompt: str, preset, api_url: str):
    """Process single query with progress tracking."""
    with st.chat_message("assistant"):
        status_container = st.status("🔄 Processing query...", expanded=True)

        with status_container:
            try:
                # Show live progress and call API
                result = show_progress_with_api_call(
                    prompt, preset, api_url, status_container
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                status_container.update(label="❌ Processing failed", state="error")
                st.stop()

        status_container.update(
            label="✅ Processing complete",
            state="complete",
            expanded=False
        )

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        metadata = result.get("metadata", {})

        st.markdown(answer, unsafe_allow_html=False)

        update_session_stats(preset.name, metadata.get("total_time_ms", 0))
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "metadata": metadata,
            "timestamp": time.time()
        })


def _run_comparison_mode(prompt: str, api_url: str):
    """Run A/B comparison with baseline and enhanced configs."""
    col1, col2 = st.columns(2)

    configs_to_compare = [
        ("baseline", PRESETS["baseline"]),
        ("enhanced_full", PRESETS["enhanced_full"]),
    ]

    results = []

    for col, (config_key, config) in zip([col1, col2], configs_to_compare):
        with col:
            with st.chat_message("assistant"):
                st.caption(f"**{config.name}**")
                status_container = st.status(f"🔄 Processing...", expanded=True)

                with status_container:
                    try:
                        # Show live progress with step-by-step tracking
                        result = show_progress_with_api_call(
                            prompt, config, api_url, status_container
                        )
                        results.append((config, result))
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        status_container.update(label="❌ Failed", state="error")
                        results.append((config, None))

                if results[-1][1]:  # If successful
                    status_container.update(label="✅ Complete", state="complete", expanded=False)
                else:
                    status_container.update(label="❌ Failed", state="error", expanded=False)

                if results[-1][1]:  # If result is not None
                    result = results[-1][1]
                    answer = result.get("answer", "")
                    metadata = result.get("metadata", {})
                    sources = result.get("sources", [])

                    # Display answer
                    st.markdown(answer, unsafe_allow_html=True)
                    if metadata:
                        _render_message_metadata(metadata)

                    # Display sources with full formatting
                    if sources:
                        # Use unique timestamp for each config
                        unique_timestamp = f"{time.time()}_{config.name}"
                        _render_sources(sources, unique_timestamp)

    # Add comparison to chat history
    if results and all(r[1] for r in results):
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**🔀 A/B Comparison Results**\n\n"
                       f"**Baseline:** {results[0][1].get('answer', '')[:100]}...\n\n"
                       f"**Enhanced:** {results[1][1].get('answer', '')[:100]}...",
            "comparison": True,
            "results": results,
            "timestamp": time.time()
        })

        # Update stats for both
        for config, result in results:
            if result:
                update_session_stats(
                    config.name,
                    result.get("metadata", {}).get("total_time_ms", 0)
                )


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    preset = render_sidebar(st.session_state.api_url)

# ============================================================================
# MAIN CHAT INTERFACE - HEADER
# ============================================================================
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.title("💬 RAGx Interactive Chat")
    st.caption(f"Using pipeline: **{preset.name}**")

with header_col2:
    st.write("")

    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if st.button("🆕", help="New chat", use_container_width=True):
            _reset_chat()
            st.rerun()

    with btn_col2:
        if st.button("🗑️", help="Delete chat", use_container_width=True):
            _reset_chat()
            st.rerun()

    with btn_col3:
        # Save button with popup
        save_clicked = st.button("💾", help="Save chat", use_container_width=True)

# Save dialog (appears below header when clicked)
if save_clicked:
    if st.session_state.messages:
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "stats": st.session_state.session_stats,
            "messages": st.session_state.messages,
        }

        save_col1, save_col2 = st.columns(2)

        with save_col1:
            st.download_button(
                label="📥 Download as JSON",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"ragx_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        with save_col2:
            md_content = f"# RAGx Chat Session\n\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

            for msg in st.session_state.messages:
                role = "**User:**" if msg["role"] == "user" else "**Assistant:**"
                md_content += f"{role}\n{msg['content']}\n\n"

                if msg["role"] == "assistant":
                    # Add sources info
                    if "sources" in msg:
                        md_content += f"*Sources: {len(msg['sources'])} documents*\n\n"

                    # Add CoVe info if present
                    if "metadata" in msg:
                        cove_data = msg["metadata"].get("cove", {})
                        if cove_data.get("needs_correction"):
                            md_content += f"*CoVe Status: {cove_data.get('status', 'N/A')} - "
                            md_content += f"{cove_data.get('num_verified', 0)} verified, "
                            md_content += f"{cove_data.get('num_refuted', 0)} refuted*\n\n"

                md_content += "---\n\n"

            st.download_button(
                label="📥 Download as Markdown",
                data=md_content,
                file_name=f"ragx_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
    else:
        st.info("💡 No messages to save yet")

st.divider()

# Display chat history
render_message_history()

# Handle example query or user input
if "example_query" in st.session_state and st.session_state.example_query:
    prompt = st.session_state.example_query
    st.session_state.example_query = None
elif prompt := st.chat_input("Ask a question..."):
    pass
else:
    prompt = None

# Process user query
if prompt:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": time.time()
    })

    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=False)

    # Check if A/B comparison mode is enabled
    if st.session_state.comparison_mode:
        _run_comparison_mode(prompt, st.session_state.api_url)
    else:
        _run_single_query(prompt, preset, st.session_state.api_url)

    st.rerun()



# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.caption("RAGx Chat UI v2.0 - Modular Edition")
st.sidebar.caption("Built with Streamlit")
