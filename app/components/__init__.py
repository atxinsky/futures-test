# coding=utf-8
"""
UI组件库
提供可复用的Streamlit组件
"""

from app.components.ui_components import (
    render_divider,
    render_metric_card,
    render_metric_row,
    render_instrument_info,
    render_page_header,
    render_status_badge,
    render_data_table,
    render_empty_state,
)

from app.components.styles import (
    get_global_css,
    THEME,
    apply_global_styles,
)

__all__ = [
    # UI组件
    'render_divider',
    'render_metric_card',
    'render_metric_row',
    'render_instrument_info',
    'render_page_header',
    'render_status_badge',
    'render_data_table',
    'render_empty_state',
    # 样式
    'get_global_css',
    'THEME',
    'apply_global_styles',
]
