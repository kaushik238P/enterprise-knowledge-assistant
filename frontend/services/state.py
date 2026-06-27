# frontend/services/state.py

import streamlit as st
from typing import Any

__all__ = [
    "init_session_state",
    "get_messages",
    "add_message",
    "clear_chat_history",
    "get_setting",
    "set_setting",
    "get_backend_status",
    "set_backend_status",
    "get_uploaded_documents",
    "set_uploaded_documents",
]

_MESSAGES_KEY = "messages"
_SETTINGS_KEY = "settings"
_BACKEND_STATUS_KEY = "backend_online"
_DOCUMENTS_KEY = "uploaded_documents"

_DEFAULT_SETTINGS: dict[str, Any] = {
    "use_agent": False,
    "use_web": False,
    "top_k": 5,
}


def init_session_state() -> None:
    """
    Initializes session state variables if they do not exist.
    """
    if _MESSAGES_KEY not in st.session_state:
        st.session_state[_MESSAGES_KEY] = []

    if _SETTINGS_KEY not in st.session_state:
        st.session_state[_SETTINGS_KEY] = _DEFAULT_SETTINGS.copy()

    if _BACKEND_STATUS_KEY not in st.session_state:
        st.session_state[_BACKEND_STATUS_KEY] = False

    if _DOCUMENTS_KEY not in st.session_state:
        st.session_state[_DOCUMENTS_KEY] = []


def get_messages() -> list[dict[str, Any]]:
    return st.session_state[_MESSAGES_KEY]


def add_message(
    role: str,
    content: str,
    **metadata: Any,
) -> None:
    message = {
    "role": role,
    "content": content,
}
    if not role.strip():
        raise ValueError("role cannot be empty.")

    if not content.strip():
        raise ValueError("content cannot be empty.")

    message.update(metadata)

    st.session_state[_MESSAGES_KEY].append(message)


def clear_chat_history() -> None:
    st.session_state[_MESSAGES_KEY] = []


def get_setting(key: str, default: Any = None) -> Any:
    return st.session_state[_SETTINGS_KEY].get(key, default)


def set_setting(key: str, value: Any) -> None:
    st.session_state[_SETTINGS_KEY][key] = value

def get_backend_status() -> bool:
    return st.session_state[_BACKEND_STATUS_KEY]


def set_backend_status(
    online: bool,
) -> None:
    st.session_state[_BACKEND_STATUS_KEY] = online
    
def get_uploaded_documents() -> list[str]:
    return st.session_state[_DOCUMENTS_KEY]


def set_uploaded_documents(
    documents: list[str],
) -> None:
    st.session_state[_DOCUMENTS_KEY] = documents