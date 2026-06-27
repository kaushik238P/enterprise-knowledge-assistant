# frontend/app.py
import streamlit as st
from frontend.services.state import init_session_state

# Initialize Session State
init_session_state()

# Register pages relative to the app.py parent directory
chat_page = st.Page("pages/chat.py", title="Chat", icon="💬", default=True)
upload_page = st.Page("pages/upload.py", title="Upload", icon="📄")

# Define navigation structure with hidden default rendering to support custom sidebar layout
pg = st.navigation([chat_page, upload_page], position="hidden")
pg.run()