# frontend/pages/chat.py

import streamlit as st

from frontend.components.chat_message import render_chat_message
from frontend.components.sidebar import render_sidebar
from frontend.services.api import APIClient
from frontend.services.state import (
    add_message,
    clear_chat_history,
    get_backend_status,
    get_messages,
    get_setting,
    init_session_state,
)

st.set_page_config(
    page_title="Chat - Enterprise Assistant",
    page_icon="💬",
    layout="wide",
)


def _get_client() -> APIClient:
    """
    Returns the API client.
    """
    return APIClient()


def _render_chat_history() -> None:
    """
    Renders the stored chat history.
    """
    for message in get_messages():
        render_chat_message(message)


def _handle_response(response: dict) -> None:
    """
    Handles the backend response.

    The /chat endpoint returns a structured ChatResponse with nested
    ``evaluation`` and ``sources`` objects. This function unpacks those
    nested fields and stores them on the message so the chat_message
    renderer can display structured data without any string parsing.
    """
    if "error" in response:
        st.error(response["error"])
        return

    evaluation: dict = response.get("evaluation") or {}
    sources: list = response.get("sources") or []

    add_message(
        role="assistant",
        content=response.get(
            "answer",
            "No response returned.",
        ),
        evaluation=evaluation,
        sources=sources,
    )

    st.rerun()


def _process_prompt(prompt: str) -> None:
    """
    Sends the user prompt to the backend.
    """
    add_message(
        role="user",
        content=prompt,
    )

    st.chat_message("user").write(prompt)

    with st.spinner("Assistant is thinking..."):
        response = _get_client().chat(
            query=prompt,
            use_agent=get_setting(
                "use_agent",
                False,
            ),
        )

    _handle_response(response)


# --------------------------------------------------------------------
# Page Initialization
# --------------------------------------------------------------------

init_session_state()

render_sidebar()

st.title("💬 Chat Assistant")

st.write(
    "Ask questions grounded in the company knowledge base."
)

if not get_backend_status():
    st.warning(
        "⚠️ The backend is currently offline."
    )

if st.button("Clear Conversation"):
    clear_chat_history()
    st.rerun()

st.markdown("---")

_render_chat_history()

if prompt := st.chat_input("Ask a question..."):
    _process_prompt(prompt)