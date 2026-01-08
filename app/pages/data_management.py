# coding=utf-8
"""
数据管理页面
期货数据下载、状态查看
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def render_data_management_page():
    """
    渲染数据管理页面

    功能：
    1. 日线数据下载
    2. 分钟数据下载
    3. 数据状态查看
    """
    st.header("数据管理")

    tab1, tab2, tab3 = st.tabs(["日线数据", "分钟数据", "数据状态"])

    with tab1:
        _render_daily_data_download()

    with tab2:
        _render_minute_data_download()

    with tab3:
        _render_data_status()


def _get_data_manager():
    """获取数据管理器相关函数"""
    try:
        from data_manager import (
            get_symbol_list_by_category,
            FUTURES_SYMBOLS,
            MINUTE_PERIODS,
            download_symbol,
            download_minute_symbol,
            get_data_status,
        )
        return {
            'get_categories': get_symbol_list_by_category,
            'FUTURES_SYMBOLS': FUTURES_SYMBOLS,
            'MINUTE_PERIODS': MINUTE_PERIODS,
            'download_symbol': download_symbol,
            'download_minute': download_minute_symbol,
            'get_status': get_data_status,
        }
    except ImportError as e:
        logger.error(f"无法导入数据管理模块: {e}")
        return None


def _render_daily_data_download():
    """渲染日线数据下载"""
    st.subheader("下载期货数据")

    dm = _get_data_manager()
    if dm is None:
        st.error("数据管理模块未加载")
        return

    categories = dm['get_categories']()
    FUTURES_SYMBOLS = dm['FUTURES_SYMBOLS']

    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox("选择类别", options=list(categories.keys()), key="daily_cat")
        symbols_in_cat = categories[category]
        selected_symbols = st.multiselect(
            "选择品种",
            options=[s[0] for s in symbols_in_cat],
            format_func=lambda x: f"{x} - {FUTURES_SYMBOLS[x][0]}",
            default=[s[0] for s in symbols_in_cat[:2]] if symbols_in_cat else [],
            key="daily_symbols"
        )

    with col2:
        st.write("**快捷选择:**")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("股指期货", key="btn_index"):
                st.session_state.quick_select = ["IF", "IH", "IC", "IM"]
                st.rerun()
        with col_b:
            if st.button("主要商品", key="btn_commodity"):
                st.session_state.quick_select = ["RB", "AU", "CU", "M", "TA"]
                st.rerun()

    # 应用快捷选择
    if 'quick_select' in st.session_state:
        selected_symbols = st.session_state.quick_select
        del st.session_state.quick_select

    if selected_symbols:
        if st.button("开始下载", type="primary", use_container_width=True, key="btn_download_daily"):
            progress_bar = st.progress(0)
            results = []

            for i, symbol in enumerate(selected_symbols):
                success, msg, count = dm['download_symbol'](symbol)
                progress_bar.progress((i + 1) / len(selected_symbols))

                if success:
                    st.write(f"✅ {msg} - {count}条")
                else:
                    st.write(f"❌ {msg}")

                results.append((symbol, success, count))

            # 下载完成统计
            success_count = sum(1 for _, s, _ in results if s)
            st.success(f"下载完成: {success_count}/{len(results)} 成功")


def _render_minute_data_download():
    """渲染分钟数据下载"""
    st.subheader("下载分钟数据")
    st.info("分钟数据来自新浪财经，约有最近1000根K线")

    dm = _get_data_manager()
    if dm is None:
        st.error("数据管理模块未加载")
        return

    categories = dm['get_categories']()
    FUTURES_SYMBOLS = dm['FUTURES_SYMBOLS']
    MINUTE_PERIODS = dm['MINUTE_PERIODS']

    col1, col2 = st.columns(2)

    with col1:
        category_min = st.selectbox("选择类别", options=list(categories.keys()), key="min_cat")
        symbols_in_cat_min = categories[category_min]
        selected_symbols_min = st.multiselect(
            "选择品种",
            options=[s[0] for s in symbols_in_cat_min],
            format_func=lambda x: f"{x} - {FUTURES_SYMBOLS[x][0]}",
            key="min_symbols"
        )

    with col2:
        selected_periods = st.multiselect(
            "K线周期",
            options=list(MINUTE_PERIODS.keys()),
            default=["60分钟"],
            key="min_periods"
        )

    if selected_symbols_min and selected_periods:
        total_tasks = len(selected_symbols_min) * len(selected_periods)
        if st.button("开始下载分钟数据", type="primary", key="btn_download_min"):
            progress_bar = st.progress(0)
            task_idx = 0

            for symbol in selected_symbols_min:
                for period_name in selected_periods:
                    period = MINUTE_PERIODS[period_name]
                    success, msg, count = dm['download_minute'](symbol, period)

                    task_idx += 1
                    progress_bar.progress(task_idx / total_tasks)

                    if success:
                        st.write(f"✅ {msg} - {count}条")
                    else:
                        st.write(f"❌ {msg}")


def _render_data_status():
    """渲染数据状态"""
    st.subheader("数据状态")

    dm = _get_data_manager()
    if dm is None:
        st.error("数据管理模块未加载")
        return

    if st.button("刷新", key="btn_refresh_status"):
        st.cache_data.clear()

    df_status = dm['get_status']()

    if df_status is None or df_status.empty:
        st.warning("暂无数据")
        return

    df_with_data = df_status[df_status['record_count'] > 0].copy()

    # 统计卡片
    col1, col2 = st.columns(2)
    with col1:
        st.metric("已有数据品种", len(df_with_data))
    with col2:
        st.metric("无数据品种", len(df_status) - len(df_with_data))

    # 数据表格
    if len(df_with_data) > 0:
        df_display = df_with_data[['symbol', 'name', 'exchange', 'start_date', 'end_date', 'record_count']].copy()
        df_display.columns = ['代码', '名称', '交易所', '起始日期', '结束日期', '数据条数']
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("暂无数据，请先下载")
