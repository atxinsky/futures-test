import streamlit as st
from app.theme import COLORS

def render_header(title: str, subtitle: str = ""):
    """
    Renders the page header using the new style.
    """
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(f"### {subtitle}")
    st.markdown("---")

def render_sidebar():
    """
    Renders the unified sidebar using native Streamlit components 
    but with custom styling applied via CSS.
    """
    with st.sidebar:
        st.markdown("## 🔮 AntiGravity Quant")
        
        # Navigation in sidebar is usually handled by page structure or radio buttons
        # Depending on how we refactor main.py, we might put the navigation selector here.
        # For now, just a placeholder / branding.
        
        st.info("System Status: Online")
        st.caption("Version 3.0.0 (Pro)")
