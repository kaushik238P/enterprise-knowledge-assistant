# frontend/pages/upload.py

import streamlit as st

from frontend.components.sidebar import render_sidebar
from frontend.services.api import APIClient
from frontend.services.state import (
    get_backend_status,
    init_session_state,
)

st.set_page_config(
    page_title="Upload - Enterprise Assistant",
    page_icon="📤",
    layout="wide",
)


def _get_client() -> APIClient:
    """
    Returns the frontend API client.
    """
    return APIClient()


def _render_uploaded_documents() -> None:
    """
    Displays the list of ingested documents and provides deletion capabilities.
    """
    st.markdown("---")
    st.subheader("📁 Document Management")

    # Fetch detailed document list
    with st.spinner("Fetching document list..."):
        detailed_docs = _get_client().list_documents_detailed()

    if not detailed_docs:
        st.info("No documents have been ingested yet.")
        return

    # Render confirmation dialog if a document is selected for deletion
    if st.session_state.get("confirm_delete_doc"):
        doc_to_delete = st.session_state.confirm_delete_doc
        st.error(
            f"🗑️ **Delete document?**\n\n"
            f"**{doc_to_delete['document_name']}**\n\n"
            f"This action is permanent. Deleted chunks cannot be recovered."
        )
        col_c1, col_c2 = st.columns([1, 8])
        if col_c1.button("Cancel", key="cancel_del_btn"):
            st.session_state.confirm_delete_doc = None
            st.rerun()
        if col_c2.button("Delete", key="confirm_del_btn", type="primary"):
            # Disable actions and show progress
            status_placeholder = st.empty()
            status_placeholder.info("Deleting document...")
            
            res = _get_client().delete_document(doc_to_delete["document_id"])
            
            if isinstance(res, dict) and "error" in res:
                status_placeholder.error(f"Error: {res['error']}")
                st.session_state.confirm_delete_doc = None
            else:
                status_placeholder.success("✓ Document deleted successfully.")
                st.session_state.confirm_delete_doc = None
                import time
                time.sleep(1.5)
                st.rerun()
        st.markdown("---")

    # Draw the table headers
    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 2, 1])
    col1.markdown("**Document Name**")
    col2.markdown("**Pages**")
    col3.markdown("**Chunks**")
    col4.markdown("**Uploaded**")
    col5.markdown("**Actions**")
    st.markdown("<hr style='margin: 0.5em 0;' />", unsafe_allow_html=True)

    # Disable delete buttons if currently in confirm/deleting state
    is_confirming = st.session_state.get("confirm_delete_doc") is not None

    for idx, doc in enumerate(detailed_docs):
        # Layout row
        col_name, col_pages, col_chunks, col_uploaded, col_action = st.columns([3, 1, 1, 2, 1])
        
        col_name.write(doc["document_name"])
        col_pages.write(str(doc["page_count"]))
        col_chunks.write(str(doc["chunk_count"]))
        
        # Ingestion timestamp
        ts_str = doc.get("ingestion_timestamp")
        if ts_str:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                uploaded_formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                uploaded_formatted = ts_str
        else:
            uploaded_formatted = "N/A"
        col_uploaded.write(uploaded_formatted)

        # Deletion action button
        # Each button needs a unique key
        btn_key = f"del_{doc['document_id']}_{idx}"
        if col_action.button("🗑 Delete", key=btn_key, disabled=is_confirming):
            st.session_state.confirm_delete_doc = doc
            st.rerun()


def _upload_documents(uploaded_files) -> None:
    """
    Uploads all selected documents.
    """
    client = _get_client()

    for uploaded_file in uploaded_files:
        with st.spinner(f"Ingesting {uploaded_file.name}..."):

            response = client.upload_document(
                file_name=uploaded_file.name,
                file_bytes=uploaded_file.read(),
            )

        if "error" in response:
            st.error(
                f"{uploaded_file.name}: {response['error']}"
            )
            continue

        st.success(
            (
                f"{uploaded_file.name} uploaded successfully.\n\n"
                f"Chunks: {response.get('chunk_count', 0)} | "
                f"Vectors: {response.get('stored_vectors', 0)}"
            )
        )


# ------------------------------------------------------------------
# Page
# ------------------------------------------------------------------

init_session_state()

render_sidebar()

st.title("📤 Document Ingestion")

st.write(
    "Upload PDF, TXT, or Markdown documents to add them to the knowledge base."
)

if not get_backend_status():
    st.error(
        "🔴 Backend is offline. Document ingestion is unavailable."
    )
else:

    uploaded_files = st.file_uploader(
        "Select documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:

        if st.button(
            "Process Selected Documents",
            type="primary",
        ):
            _upload_documents(uploaded_files)

_render_uploaded_documents()