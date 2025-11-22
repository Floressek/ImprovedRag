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
        # Special rendering for comparison messages
        if message.get("comparison"):
            _render_comparison_message(message)
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show metadata for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    _render_message_metadata(message["metadata"])

                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    _render_sources(message["sources"], message.get("timestamp", ""))


def _render_comparison_message(message: Dict[str, Any]):
    """Render A/B comparison results side-by-side with full formatting."""
    st.markdown("### 🔀 A/B Comparison Results")

    results = message.get("results", [])
    if not results or len(results) != 2:
        st.error("Invalid comparison data")
        return

    col1, col2 = st.columns(2)

    for col, (config, result) in zip([col1, col2], results):
        with col:
            with st.chat_message("assistant"):
                st.caption(f"**{config.name}**")

                if result is None:
                    st.error("❌ Failed to process")
                    continue

                # Display answer
                answer = result.get("answer", "")
                st.markdown(answer)

                # Display metadata
                metadata = result.get("metadata", {})
                if metadata:
                    _render_message_metadata(metadata)

                # Display sources
                sources = result.get("sources", [])
                if sources:
                    timestamp = message.get("timestamp", "")
                    # Add config name to timestamp to ensure unique keys
                    unique_timestamp = f"{timestamp}_{config.name}"
                    _render_sources(sources, unique_timestamp)


def _render_message_metadata(metadata: Dict[str, Any]):
    """Render timing and pipeline metadata for a message."""

    # CoVe status alert (if corrections were made)
    cove_data = metadata.get("cove", {})
    if cove_data.get("needs_correction"):
        status = cove_data.get("status", "UNKNOWN")
        num_refuted = cove_data.get("num_refuted", 0)
        num_insufficient = cove_data.get("num_insufficient", 0)

        if num_refuted > 0 or num_insufficient > 0:
            st.warning(
                f"⚠️ CoVe detected {num_refuted + num_insufficient} issue(s) and applied corrections "
                f"(Status: {status})"
            )
        else:
            st.info("ℹ️ CoVe made improvements to the answer")

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

    # CoVe verification details (if available)
    if cove_data:
        with st.expander("🔍 CoVe Verification Details"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Claims Verified", cove_data.get("num_verified", 0))
                st.metric("Refuted", cove_data.get("num_refuted", 0))

            with col2:
                st.metric("Insufficient Evidence", cove_data.get("num_insufficient", 0))
                st.metric("Missing Citations", cove_data.get("missing_citations", 0))

            with col3:
                recovery_attempted = cove_data.get("recovery_attempted", False)
                recovery_helped = cove_data.get("recovery_helped", False)
                st.write(f"**Recovery:** {'✅ Helped' if recovery_helped else ('⚠️ Tried' if recovery_attempted else '➖ N/A')}")
                st.write(f"**Status:** {cove_data.get('status', 'N/A')}")

            # Show corrections if made
            correction_meta = cove_data.get("correction_metadata", {})
            if correction_meta:
                st.markdown("**Corrections Applied:**")
                if correction_meta.get("citations_injected"):
                    st.success("✅ Citations injected for verified claims")
                if correction_meta.get("post_correction_citations_injected"):
                    st.success("✅ Post-correction citations added")
                if correction_meta.get("applied"):
                    st.warning(f"⚠️ {correction_meta.get('applied_count', 0)} suggestions applied")


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
