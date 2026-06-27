# frontend/pages/settings.py

import streamlit as st
from frontend.services.state import init_session_state, set_setting, get_setting
from frontend.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Settings - Enterprise Assistant",
    page_icon="⚙️",
    layout="wide",
)

init_session_state()
render_sidebar()

st.title("⚙️ System Settings")
st.write("Configure the RAG and retrieval pipeline parameters.")

st.markdown("---")

# Slider for Top-K retrieval
top_k = st.slider(
    "Top-K Document Retrieval",
    min_value=1,
    max_value=20,
    value=get_setting("top_k", 5),
    help="Number of document chunks to retrieve and feed into the LLM context.",
)
set_setting("top_k", top_k)

st.markdown("---")
st.subheader("Workflow Configuration")

# Toggle for Agent Mode
use_agent = st.checkbox(
    "Enable Agent Mode (LangGraph Router)",
    value=get_setting("use_agent", False),
    help="Routes the query dynamically between local documents, web search, or a hybrid combination.",
)
set_setting("use_agent", use_agent)

# Toggle for Web Search
use_web = st.checkbox(
    "Enable Web Search (Tavily)",
    value=get_setting("use_web", False),
    help="Allows search to fallback to the internet using Tavily.",
)
set_setting("use_web", use_web)

st.success("Settings updated successfully.")