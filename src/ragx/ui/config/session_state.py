"""Session state initialization for Streamlit."""

import streamlit as st


def initialize_session_state():
    """Initialize all session state variables."""

    # Chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # API configuration
    if "api_url" not in st.session_state:
        st.session_state.api_url = "http://localhost:8080"

    # UI modes
    if "comparison_mode" not in st.session_state:
        st.session_state.comparison_mode = False

    # Session statistics
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = {
            "total_queries": 0,
            "total_time_ms": 0,
            "configs_used": {},
        }

    # Example query buffer
    if "example_query" not in st.session_state:
        st.session_state.example_query = None

    # Connection status cache
    if "connection_status" not in st.session_state:
        st.session_state.connection_status = {
            "status": None,   # "ok", "error", "unknown"
            "message": "Not checked yet",
        }
