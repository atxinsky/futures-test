# coding=utf-8
"""
CSS样式集中管理
统一定义主题色、组件样式、布局样式
"""

import streamlit as st
from typing import Dict, Any


# ============== 主题配置 ==============

THEME: Dict[str, Any] = {
    # 主色调
    'primary': '#1f77b4',
    'primary_dark': '#1a5f8a',
    'primary_light': '#4a9fd4',

    # 状态色
    'success': '#2ecc71',
    'success_dark': '#27ae60',
    'warning': '#f39c12',
    'warning_dark': '#d68910',
    'danger': '#e74c3c',
    'danger_dark': '#c0392b',
    'info': '#3498db',

    # 中性色
    'text_primary': '#2c3e50',
    'text_secondary': '#7f8c8d',
    'text_muted': '#95a5a6',
    'background': '#ffffff',
    'background_secondary': '#f8f9fa',
    'border': '#dee2e6',

    # 盈亏色
    'profit': '#e74c3c',      # 盈利红色（中国习惯）
    'loss': '#27ae60',        # 亏损绿色
    'profit_alt': '#27ae60',  # 盈利绿色（国际习惯）
    'loss_alt': '#e74c3c',    # 亏损红色

    # 间距
    'spacing_xs': '4px',
    'spacing_sm': '8px',
    'spacing_md': '16px',
    'spacing_lg': '24px',
    'spacing_xl': '32px',

    # 圆角
    'radius_sm': '4px',
    'radius_md': '8px',
    'radius_lg': '12px',

    # 阴影
    'shadow_sm': '0 1px 3px rgba(0,0,0,0.1)',
    'shadow_md': '0 4px 6px rgba(0,0,0,0.1)',
    'shadow_lg': '0 10px 15px rgba(0,0,0,0.1)',
}


def get_global_css() -> str:
    """
    获取全局CSS样式

    包含：
    - 指标卡片样式
    - 数据表格样式
    - 状态标签样式
    - 布局辅助类
    """
    return f"""
    <style>
    /* ============== 全局样式 ============== */

    /* 隐藏Streamlit默认的页脚和菜单 */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* 主标题样式 */
    .main-title {{
        font-size: 2rem;
        font-weight: 700;
        color: {THEME['text_primary']};
        margin-bottom: {THEME['spacing_lg']};
        padding-bottom: {THEME['spacing_md']};
        border-bottom: 3px solid {THEME['primary']};
    }}

    /* 子标题样式 */
    .sub-title {{
        font-size: 1.25rem;
        font-weight: 600;
        color: {THEME['text_primary']};
        margin: {THEME['spacing_md']} 0;
    }}

    /* ============== 指标卡片样式 ============== */

    .metric-card {{
        background: linear-gradient(135deg, {THEME['background']} 0%, {THEME['background_secondary']} 100%);
        border-radius: {THEME['radius_md']};
        padding: {THEME['spacing_md']};
        border: 1px solid {THEME['border']};
        box-shadow: {THEME['shadow_sm']};
        transition: transform 0.2s, box-shadow 0.2s;
    }}

    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: {THEME['shadow_md']};
    }}

    .metric-card-primary {{
        background: linear-gradient(135deg, {THEME['primary']} 0%, {THEME['primary_dark']} 100%);
        color: white;
    }}

    .metric-card-success {{
        background: linear-gradient(135deg, {THEME['success']} 0%, {THEME['success_dark']} 100%);
        color: white;
    }}

    .metric-card-danger {{
        background: linear-gradient(135deg, {THEME['danger']} 0%, {THEME['danger_dark']} 100%);
        color: white;
    }}

    .metric-label {{
        font-size: 0.875rem;
        color: {THEME['text_secondary']};
        margin-bottom: {THEME['spacing_xs']};
    }}

    .metric-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {THEME['text_primary']};
    }}

    .metric-delta {{
        font-size: 0.875rem;
        margin-top: {THEME['spacing_xs']};
    }}

    .metric-delta-positive {{
        color: {THEME['profit']};
    }}

    .metric-delta-negative {{
        color: {THEME['loss']};
    }}

    /* ============== 状态标签样式 ============== */

    .status-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }}

    .status-success {{
        background-color: rgba(46, 204, 113, 0.15);
        color: {THEME['success_dark']};
    }}

    .status-warning {{
        background-color: rgba(243, 156, 18, 0.15);
        color: {THEME['warning_dark']};
    }}

    .status-danger {{
        background-color: rgba(231, 76, 60, 0.15);
        color: {THEME['danger_dark']};
    }}

    .status-info {{
        background-color: rgba(52, 152, 219, 0.15);
        color: {THEME['info']};
    }}

    .status-neutral {{
        background-color: rgba(127, 140, 141, 0.15);
        color: {THEME['text_secondary']};
    }}

    /* ============== 数据表格样式 ============== */

    .dataframe {{
        font-size: 0.875rem;
    }}

    .dataframe th {{
        background-color: {THEME['background_secondary']} !important;
        color: {THEME['text_primary']} !important;
        font-weight: 600 !important;
        padding: {THEME['spacing_sm']} {THEME['spacing_md']} !important;
    }}

    .dataframe td {{
        padding: {THEME['spacing_sm']} {THEME['spacing_md']} !important;
    }}

    /* 盈亏着色 */
    .pnl-positive {{
        color: {THEME['profit']} !important;
        font-weight: 600;
    }}

    .pnl-negative {{
        color: {THEME['loss']} !important;
        font-weight: 600;
    }}

    /* ============== 分隔线样式 ============== */

    .divider {{
        height: 1px;
        background: linear-gradient(to right, transparent, {THEME['border']}, transparent);
        margin: {THEME['spacing_lg']} 0;
    }}

    .divider-thick {{
        height: 2px;
        background: {THEME['primary']};
        margin: {THEME['spacing_xl']} 0;
    }}

    /* ============== 卡片容器样式 ============== */

    .card-container {{
        background: {THEME['background']};
        border-radius: {THEME['radius_lg']};
        padding: {THEME['spacing_lg']};
        border: 1px solid {THEME['border']};
        box-shadow: {THEME['shadow_sm']};
        margin-bottom: {THEME['spacing_md']};
    }}

    .card-header {{
        font-size: 1.125rem;
        font-weight: 600;
        color: {THEME['text_primary']};
        margin-bottom: {THEME['spacing_md']};
        padding-bottom: {THEME['spacing_sm']};
        border-bottom: 2px solid {THEME['primary']};
    }}

    /* ============== 空状态样式 ============== */

    .empty-state {{
        text-align: center;
        padding: {THEME['spacing_xl']};
        color: {THEME['text_muted']};
    }}

    .empty-state-icon {{
        font-size: 3rem;
        margin-bottom: {THEME['spacing_md']};
    }}

    .empty-state-text {{
        font-size: 1rem;
        margin-bottom: {THEME['spacing_md']};
    }}

    /* ============== 品种信息卡片样式 ============== */

    .instrument-info {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: {THEME['spacing_md']};
        padding: {THEME['spacing_md']};
        background: {THEME['background_secondary']};
        border-radius: {THEME['radius_md']};
    }}

    .instrument-info-item {{
        text-align: center;
    }}

    .instrument-info-label {{
        font-size: 0.75rem;
        color: {THEME['text_secondary']};
        margin-bottom: {THEME['spacing_xs']};
    }}

    .instrument-info-value {{
        font-size: 1rem;
        font-weight: 600;
        color: {THEME['text_primary']};
    }}

    /* ============== 按钮样式优化 ============== */

    .stButton > button {{
        border-radius: {THEME['radius_md']};
        font-weight: 500;
        transition: all 0.2s;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: {THEME['shadow_sm']};
    }}

    /* 主要按钮 */
    .stButton > button[kind="primary"] {{
        background-color: {THEME['primary']};
    }}

    /* ============== Expander样式优化 ============== */

    .streamlit-expanderHeader {{
        font-weight: 600;
        color: {THEME['text_primary']};
    }}

    /* ============== Tab样式优化 ============== */

    .stTabs [data-baseweb="tab-list"] {{
        gap: {THEME['spacing_sm']};
    }}

    .stTabs [data-baseweb="tab"] {{
        padding: {THEME['spacing_sm']} {THEME['spacing_lg']};
        border-radius: {THEME['radius_md']} {THEME['radius_md']} 0 0;
    }}

    /* ============== 侧边栏样式 ============== */

    .css-1d391kg {{
        padding: {THEME['spacing_lg']};
    }}

    /* ============== 响应式布局 ============== */

    @media (max-width: 768px) {{
        .metric-card {{
            padding: {THEME['spacing_sm']};
        }}

        .metric-value {{
            font-size: 1.25rem;
        }}

        .instrument-info {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}
    </style>
    """


def apply_global_styles():
    """应用全局样式到Streamlit页面"""
    st.markdown(get_global_css(), unsafe_allow_html=True)


def get_pnl_color(value: float, use_chinese_convention: bool = True) -> str:
    """
    获取盈亏颜色

    Args:
        value: 盈亏值
        use_chinese_convention: 是否使用中国习惯（红涨绿跌）

    Returns:
        颜色代码
    """
    if use_chinese_convention:
        return THEME['profit'] if value > 0 else THEME['loss'] if value < 0 else THEME['text_secondary']
    else:
        return THEME['profit_alt'] if value > 0 else THEME['loss_alt'] if value < 0 else THEME['text_secondary']


def get_status_class(status: str) -> str:
    """
    获取状态对应的CSS类

    Args:
        status: 状态字符串 ('success', 'warning', 'danger', 'info', 'neutral')

    Returns:
        CSS类名
    """
    status_map = {
        'success': 'status-success',
        'warning': 'status-warning',
        'danger': 'status-danger',
        'info': 'status-info',
        'neutral': 'status-neutral',
        # 中文映射
        '成功': 'status-success',
        '警告': 'status-warning',
        '错误': 'status-danger',
        '信息': 'status-info',
        '运行中': 'status-success',
        '已停止': 'status-neutral',
        '暂停': 'status-warning',
    }
    return status_map.get(status, 'status-neutral')
