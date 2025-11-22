"""Progress tracking component with realistic timing."""

import time
import threading
import streamlit as st
from typing import Dict, Any, Callable

from ..types import PipelineConfig
from ..helpers import get_pipeline_steps, call_rag_api


def show_progress_with_api_call(
    prompt: str,
    config: PipelineConfig,
    api_url: str,
    status_container
) -> Dict[str, Any]:
    """
    Show live progress bar while API call executes.

    Uses threading to run API call in background while updating progress
    based on realistic step duration estimates.

    Args:
        prompt: User query
        config: Pipeline configuration
        api_url: API base URL
        status_container: Streamlit status container

    Returns:
        API response dict

    Raises:
        Exception: If API call fails or times out
    """
    steps = get_pipeline_steps(config)
    enabled_steps = [s for s in steps if s.enabled]
    total_steps = len(enabled_steps)

    # Calculate total estimated time
    total_estimated_time = sum(s.estimated_duration for s in enabled_steps)

    start_time = time.time()

    # Show progress bar
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    timing_text = st.empty()

    # Thread communication
    api_done = threading.Event()
    api_result = {}
    api_error = {}

    def api_call_thread():
        """Background thread for API call."""
        try:
            result = call_rag_api(prompt, config, api_url)
            api_result['data'] = result
        except Exception as e:
            api_error['error'] = e
        finally:
            api_done.set()

    # Start API call in background
    thread = threading.Thread(target=api_call_thread, daemon=True)
    thread.start()

    # Show progress based on realistic timing estimates
    cumulative_time = 0.0
    for idx, step in enumerate(steps):
        if step.enabled:
            # Calculate progress percentage based on time estimates
            progress = cumulative_time / total_estimated_time if total_estimated_time > 0 else 0
            progress = min(progress, 0.95)  # Cap at 95% until actually done

            progress_bar.progress(progress)
            status_text.markdown(step.message)

            elapsed = time.time() - start_time
            timing_text.caption(f"⏱️ Elapsed: {elapsed:.1f}s / ~{total_estimated_time:.1f}s")

            # Wait for this step's estimated duration (in small chunks to stay responsive)
            step_wait_time = step.estimated_duration
            intervals = int(step_wait_time / 0.2)  # Check every 200ms

            for _ in range(intervals):
                if api_done.is_set():
                    break
                time.sleep(0.2)

            cumulative_time += step.estimated_duration

            if api_done.is_set():
                break

    # Wait for API to complete (with timeout)
    api_done.wait(timeout=120)

    # Check for errors
    if 'error' in api_error:
        raise api_error['error']

    if 'data' not in api_result:
        raise TimeoutError("API request timed out after 120 seconds")

    result = api_result['data']

    # Complete progress
    progress_bar.progress(1.0)
    total_time = (time.time() - start_time) * 1000
    status_text.markdown(f"✨ **Complete!** Total: {total_time:.0f}ms")
    timing_text.caption(f"⏱️ Actual: {total_time/1000:.1f}s (estimated: {total_estimated_time:.1f}s)")

    return result
