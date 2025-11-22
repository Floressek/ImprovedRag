"""Chat display components for messages and sources."""

import streamlit as st
from typing import Dict, Any, List

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def render_message_history():
    """Render all messages in chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                _render_message_metadata(message["metadata"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                _render_sources(message["sources"], message.get("timestamp", ""))


def _render_message_metadata(metadata: Dict[str, Any]):
    """Render timing and pipeline metadata for a message."""

    # Timing details with chart
    with st.expander("⏱️ Timing Details"):
        timings = metadata.get("timings", {})

        # Metrics row
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

        # Timing breakdown chart
        st.markdown("**Breakdown:**")
        timing_data = {
            "Query Analysis": timings.get('rewrite_time_ms', 0),
            "Retrieval": timings.get('retrieval_time_ms', 0),
            "Reranking": timings.get('rerank_time_ms', 0),
            "Generation": timings.get('llm_time_ms', 0),
            "CoVe": timings.get('cove_time_ms', 0),
        }
        # Filter out zero values
        timing_data = {k: v for k, v in timing_data.items() if v > 0}

        if timing_data and PLOTLY_AVAILABLE:
            fig = px.bar(
                x=list(timing_data.values()),
                y=list(timing_data.keys()),
                orientation='h',
                labels={'x': 'Time (ms)', 'y': 'Phase'},
                title='Pipeline Phase Timings'
            )
            fig.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        elif timing_data and not PLOTLY_AVAILABLE:
            # Fallback: show as text
            for phase, ms in timing_data.items():
                st.write(f"  • {phase}: {ms:.0f}ms")

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


def _render_sources(sources: List[Dict[str, Any]], timestamp: str):
    """Render source citations with metadata."""
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for i, source in enumerate(sources, 1):
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

            # Full text display
            source_text = source.get('text', '')
            if len(source_text) > 300:
                # Preview
                st.caption(source_text[:300] + "...")

                # Toggle to show full text
                show_full = st.checkbox(
                    "📖 View full text",
                    key=f"show_full_{i}_{timestamp}"
                )
                if show_full:
                    st.text_area(
                        "Source text",
                        source_text,
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"source_{i}_{timestamp}"
                    )
                    st.caption("💡 Tip: Select text above and Ctrl+C to copy")
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
