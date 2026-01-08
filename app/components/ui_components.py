# coding=utf-8
"""
可复用UI组件库
提供统一的Streamlit组件封装
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from app.components.styles import THEME, get_pnl_color, get_status_class


def render_divider(thick: bool = False):
    """
    渲染分隔线

    Args:
        thick: 是否使用粗分隔线
    """
    css_class = "divider-thick" if thick else "divider"
    st.markdown(f'<div class="{css_class}"></div>', unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: Any,
    delta: Any = None,
    delta_color: str = "normal",
    style: str = "default",
    help_text: str = None
):
    """
    渲染指标卡片

    Args:
        label: 指标标签
        value: 指标值
        delta: 变化值（可选）
        delta_color: 变化颜色 ('normal', 'inverse', 'off')
        style: 卡片样式 ('default', 'primary', 'success', 'danger')
        help_text: 帮助文本
    """
    # 使用Streamlit原生metric组件
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )


def render_metric_row(
    metrics: List[Dict[str, Any]],
    columns: int = None
):
    """
    渲染一行指标卡片

    Args:
        metrics: 指标列表，每个元素包含 {label, value, delta?, delta_color?, help?}
        columns: 列数（默认与指标数相同）

    Example:
        render_metric_row([
            {'label': '总收益', 'value': '12,345', 'delta': '+5.2%'},
            {'label': '胜率', 'value': '65%'},
            {'label': '最大回撤', 'value': '-8.5%', 'delta_color': 'inverse'},
        ])
    """
    num_metrics = len(metrics)
    num_cols = columns or num_metrics

    cols = st.columns(num_cols)

    for i, metric in enumerate(metrics):
        with cols[i % num_cols]:
            st.metric(
                label=metric.get('label', ''),
                value=metric.get('value', ''),
                delta=metric.get('delta'),
                delta_color=metric.get('delta_color', 'normal'),
                help=metric.get('help')
            )


def render_instrument_info(
    symbol: str,
    name: str = None,
    exchange: str = None,
    multiplier: float = None,
    margin_rate: float = None,
    tick_size: float = None,
    min_volume: int = None
):
    """
    渲染品种信息卡片

    Args:
        symbol: 品种代码
        name: 品种名称
        exchange: 交易所
        multiplier: 合约乘数
        margin_rate: 保证金率
        tick_size: 最小变动价位
        min_volume: 最小交易手数
    """
    # 尝试从配置获取品种信息
    if not all([name, exchange, multiplier]):
        try:
            from config import get_instrument
            inst = get_instrument(symbol)
            if inst:
                name = name or inst.get('name', symbol)
                exchange = exchange or inst.get('exchange', '-')
                multiplier = multiplier or inst.get('multiplier', 0)
                margin_rate = margin_rate or inst.get('margin_rate', 0)
                tick_size = tick_size or inst.get('tick_size', 0)
                min_volume = min_volume or inst.get('min_volume', 1)
        except ImportError:
            pass

    # 渲染6列布局
    cols = st.columns(6)

    info_items = [
        ("品种代码", symbol),
        ("品种名称", name or symbol),
        ("交易所", exchange or "-"),
        ("合约乘数", f"{multiplier}" if multiplier else "-"),
        ("保证金率", f"{margin_rate*100:.1f}%" if margin_rate else "-"),
        ("最小变动", f"{tick_size}" if tick_size else "-"),
    ]

    for col, (label, value) in zip(cols, info_items):
        with col:
            st.metric(label=label, value=value)


def render_page_header(
    title: str,
    subtitle: str = None,
    icon: str = None
):
    """
    渲染页面头部

    Args:
        title: 页面标题
        subtitle: 副标题（可选）
        icon: 图标（可选，emoji）
    """
    header_text = f"{icon} {title}" if icon else title
    st.markdown(f'<h1 class="main-title">{header_text}</h1>', unsafe_allow_html=True)

    if subtitle:
        st.markdown(f'<p class="sub-title">{subtitle}</p>', unsafe_allow_html=True)


def render_status_badge(
    status: str,
    text: str = None
) -> str:
    """
    渲染状态标签

    Args:
        status: 状态类型 ('success', 'warning', 'danger', 'info', 'neutral')
        text: 显示文本（默认使用status）

    Returns:
        HTML字符串
    """
    css_class = get_status_class(status)
    display_text = text or status

    html = f'<span class="status-badge {css_class}">{display_text}</span>'
    st.markdown(html, unsafe_allow_html=True)
    return html


def render_data_table(
    df: pd.DataFrame,
    height: int = None,
    hide_index: bool = True,
    column_config: Dict = None,
    pnl_columns: List[str] = None,
    use_container_width: bool = True
):
    """
    渲染数据表格

    Args:
        df: DataFrame数据
        height: 表格高度（像素）
        hide_index: 是否隐藏索引
        column_config: 列配置
        pnl_columns: 需要盈亏着色的列名列表
        use_container_width: 是否使用容器宽度
    """
    if df is None or df.empty:
        render_empty_state("暂无数据")
        return

    # 处理盈亏着色
    styled_df = df.copy()

    if pnl_columns:
        def color_pnl(val):
            if isinstance(val, (int, float)):
                color = get_pnl_color(val)
                return f'color: {color}'
            return ''

        for col in pnl_columns:
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].apply(
                    lambda x: f"{x:+.2f}" if isinstance(x, (int, float)) else x
                )

    # 使用st.dataframe
    st.dataframe(
        styled_df,
        height=height,
        hide_index=hide_index,
        column_config=column_config,
        use_container_width=use_container_width
    )


def render_empty_state(
    message: str = "暂无数据",
    icon: str = "📭",
    action_label: str = None,
    action_callback = None
):
    """
    渲染空状态

    Args:
        message: 提示消息
        icon: 图标（emoji）
        action_label: 操作按钮文本
        action_callback: 操作按钮回调
    """
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-text">{message}</div>
    </div>
    """, unsafe_allow_html=True)

    if action_label and action_callback:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(action_label, use_container_width=True):
                action_callback()


def render_card(
    title: str,
    content_func,
    expanded: bool = True
):
    """
    渲染卡片容器

    Args:
        title: 卡片标题
        content_func: 内容渲染函数
        expanded: 是否默认展开
    """
    with st.expander(title, expanded=expanded):
        content_func()


def render_tabs_container(
    tabs: List[str],
    content_funcs: List
):
    """
    渲染Tab容器

    Args:
        tabs: Tab标签列表
        content_funcs: 各Tab的内容渲染函数列表
    """
    tab_objects = st.tabs(tabs)

    for tab, func in zip(tab_objects, content_funcs):
        with tab:
            func()


def render_two_column_layout(
    left_func,
    right_func,
    left_width: float = 0.5,
    gap: str = "medium"
):
    """
    渲染两列布局

    Args:
        left_func: 左列渲染函数
        right_func: 右列渲染函数
        left_width: 左列宽度比例 (0-1)
        gap: 间距 ('small', 'medium', 'large')
    """
    gap_map = {'small': 'small', 'medium': 'medium', 'large': 'large'}
    right_width = 1 - left_width

    col1, col2 = st.columns([left_width, right_width], gap=gap_map.get(gap, 'medium'))

    with col1:
        left_func()

    with col2:
        right_func()


def render_three_column_layout(
    left_func,
    mid_func,
    right_func,
    widths: tuple = (1, 1.5, 0.8),
    gap: str = "medium"
):
    """
    渲染三列布局（用于回测配置页面）

    Args:
        left_func: 左列渲染函数
        mid_func: 中列渲染函数
        right_func: 右列渲染函数
        widths: 列宽度比例
        gap: 间距
    """
    col1, col2, col3 = st.columns(list(widths), gap=gap)

    with col1:
        left_func()

    with col2:
        mid_func()

    with col3:
        right_func()


def render_confirm_dialog(
    title: str,
    message: str,
    confirm_label: str = "确认",
    cancel_label: str = "取消",
    on_confirm = None,
    on_cancel = None,
    danger: bool = False
):
    """
    渲染确认对话框

    Args:
        title: 对话框标题
        message: 确认消息
        confirm_label: 确认按钮文本
        cancel_label: 取消按钮文本
        on_confirm: 确认回调
        on_cancel: 取消回调
        danger: 是否为危险操作
    """
    st.warning(f"**{title}**\n\n{message}")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        button_type = "primary" if not danger else "secondary"
        if st.button(cancel_label, use_container_width=True):
            if on_cancel:
                on_cancel()

    with col2:
        if st.button(confirm_label, type="primary" if not danger else "secondary", use_container_width=True):
            if on_confirm:
                on_confirm()


def render_progress_info(
    current: int,
    total: int,
    label: str = "进度",
    show_percentage: bool = True
):
    """
    渲染进度信息

    Args:
        current: 当前值
        total: 总数
        label: 标签
        show_percentage: 是否显示百分比
    """
    progress = current / total if total > 0 else 0

    if show_percentage:
        st.progress(progress, text=f"{label}: {current}/{total} ({progress*100:.1f}%)")
    else:
        st.progress(progress, text=f"{label}: {current}/{total}")


def render_timestamp(
    dt: datetime = None,
    format: str = "%Y-%m-%d %H:%M:%S",
    label: str = "更新时间"
):
    """
    渲染时间戳

    Args:
        dt: datetime对象（默认当前时间）
        format: 时间格式
        label: 标签
    """
    if dt is None:
        dt = datetime.now()

    st.caption(f"{label}: {dt.strftime(format)}")
