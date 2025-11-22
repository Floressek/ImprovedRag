import streamlit as st
import time

from src.ragx.ui.config import PRESETS, initialize_session_state
from src.ragx.ui.components import (
    render_sidebar,
    render_message_history,
    show_progress_with_api_call,
)
from src.ragx.ui.helpers.helpers import call_rag_api, update_session_stats

# Page configuration
st.set_page_config(
    page_title="RAGx Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
initialize_session_state()


# ============================================================================
# HELPER FUNCTIONS FOR QUERY PROCESSING
# ============================================================================

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

        # Update status to complete
        status_container.update(
            label="✅ Processing complete",
            state="complete",
            expanded=False
        )

        # Display answer
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        metadata = result.get("metadata", {})

        st.markdown(answer)

        # Update session stats
        update_session_stats(preset.name, metadata.get("total_time_ms", 0))

        # Add to chat history
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
                status = st.status(f"🔄 Processing...", expanded=True)

                with status:
                    try:
                        st.write("📡 Calling API...")
                        result = call_rag_api(prompt, config, api_url)
                        st.write(f"✨ Done! {result.get('metadata', {}).get('total_time_ms', 0):.0f}ms")
                        results.append((config, result))
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        status.update(label="❌ Failed", state="error")
                        results.append((config, None))

                status.update(label="✅ Complete", state="complete", expanded=False)

                if results[-1][1]:  # If result is not None
                    answer = results[-1][1].get("answer", "")
                    metadata = results[-1][1].get("metadata", {})

                    st.markdown(answer)

                    # Quick stats
                    st.caption(
                        f"⏱️ {metadata.get('total_time_ms', 0):.0f}ms | "
                        f"📚 {metadata.get('num_sources', 0)} sources | "
                        f"{'🔀 Multihop' if metadata.get('is_multihop') else '📄 Single'}"
                    )

    # Add comparison to chat history
    if all(r[1] for r in results):
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
# MAIN CHAT INTERFACE
# ============================================================================

st.title("💬 RAGx Interactive Chat")
st.caption(f"Using pipeline: **{preset.name}**")

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
        st.markdown(prompt)

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
