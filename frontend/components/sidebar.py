# frontend/components/sidebar.py

import streamlit as st

from frontend.services.api import APIClient
from frontend.services.state import (
    get_setting,
    set_setting,
    set_backend_status,
)

__all__ = [
    "render_sidebar",
]


def _get_client() -> APIClient:
    """
    Returns the frontend API client.
    """
    return APIClient()


def _check_backend() -> bool:
    """
    Checks whether the backend API is available.
    """
    return _get_client().health()


def _render_backend_status(
    online: bool,
) -> None:
    """
    Displays the backend status.
    """
    if online:
        st.success("🟢 System Online")
    else:
        st.error("🔴 System Offline")


def _render_settings() -> None:
    """
    Renders the global application settings.
    """
    st.subheader("Global Controls")

    use_agent = st.toggle(
        "Enable Agent Mode",
        value=get_setting("use_agent", False),
        help=(
            "Route queries through the LangGraph agent "
            "instead of the classic RAG pipeline."
        ),
    )
    set_setting("use_agent", use_agent)

    use_web = st.toggle(
        "Enable Web Search",
        value=get_setting("use_web", False),
        help=(
            "Enable Tavily web search for real-time information."
        ),
    )
    set_setting("use_web", use_web)


def render_sidebar() -> None:
    """
    Renders the application sidebar.
    """
    backend_online = _check_backend()
    set_backend_status(backend_online)

    with st.sidebar:
        st.title("🛡️ Enterprise Assistant")

        st.markdown("---")

        _render_backend_status(backend_online)

        st.markdown("---")

        _render_settings()

        st.markdown("---")

        st.subheader("Navigation")
        st.page_link("pages/chat.py", label="Chat", icon="💬")
        st.page_link("pages/upload.py", label="Upload", icon="📄")

        st.markdown("---")

        st.markdown(
            (
                "<div style='text-align:center;"
                "color:gray;"
                "font-size:11px;'>"
                "Enterprise Knowledge Assistant v1.0.0"
                "</div>"
            ),
            unsafe_allow_html=True,
        )