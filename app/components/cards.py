import streamlit as st
import pandas as pd
from app.theme import COLORS

def render_metric_card(label: str, value: str, delta: str = None, help_text: str = None):
    """
    Renders a metric card with optional delta and help text.
    Uses custom CSS class 'metric-card' for styling.
    """
    delta_html = ""
    if delta:
        # Determine delta color class
        clean_delta = delta.replace("%", "").replace("+", "").replace("-", "")
        try:
            val = float(clean_delta)
            if "+" in delta or (delta[0] != "-" and val >= 0):  # Assume positive if not explicitly negative
                 delta_class = "delta-positive"
                 icon = "▲"
            elif "-" in delta:
                delta_class = "delta-negative"
                icon = "▼"
            else:
                delta_class = "delta-neutral"
                icon = "•"
        except ValueError:
             delta_class = "delta-neutral"
             icon = "•"
             
        delta_html = f'<div class="metric-delta {delta_class}">{icon} {delta}</div>'
    
    help_html = f'<div title="{help_text}" style="cursor:help; font-size:0.8em">ℹ️</div>' if help_text else ""
    
    html = f"""
    <div class="metric-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
             <div class="metric-label">{label}</div>
             {help_html}
        </div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_status_card(title: str, status: str, details: str = ""):
    """
    Renders a status card.
    Status can be: 'active', 'inactive', 'warning', 'error'
    """
    status_map = {
        "active": {"class": "status-active", "text": "运行中"},
        "inactive": {"class": "status-inactive", "text": "已停止"},
        "warning": {"class": "status-warning", "text": "警告"},
        "error": {"class": "status-error", "text": "错误"}
    }
    
    s = status_map.get(status, status_map["inactive"])
    
    html = f"""
    <div class="status-card">
        <div style="font-weight:600; color:var(--text-primary);">
            {title}
        </div>
        <div style="display:flex; align-items:center;">
            <div class="status-indicator {s['class']}"></div>
            <span style="font-size:0.9em; color:var(--text-secondary);">{s['text']}</span>
        </div>
    </div>
    """
    if details:
        html += f'<div style="font-size:0.8em; color:var(--text-secondary); margin-top:4px; padding-left:4px;">{details}</div>'
        
    st.markdown(html, unsafe_allow_html=True)
