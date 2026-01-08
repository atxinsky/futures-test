# coding=utf-8
"""
仪表盘页面
系统概览、快速入口、状态监控
"""

import streamlit as st
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def render_dashboard_page():
    """
    渲染仪表盘页面

    显示：
    1. 系统状态概览（模拟/实盘运行状态）
    2. 快速操作入口
    3. 使用流程说明
    """
    st.title("系统概览")

    # 获取引擎状态
    sim_engine = st.session_state.get('sim_engine')
    live_engine = st.session_state.get('live_engine')

    sim_running = sim_engine is not None and getattr(sim_engine, 'is_running', False)
    live_running = live_engine is not None and getattr(live_engine, 'is_running', False)

    # 导入策略和品种配置
    try:
        from strategies import get_all_strategies
        from config import INSTRUMENTS
        strategy_count = len(get_all_strategies())
        instrument_count = len(INSTRUMENTS)
    except ImportError:
        strategy_count = 0
        instrument_count = 0

    # ============ 系统状态卡片 ============
    col1, col2, col3 = st.columns(3)

    with col1:
        _render_sim_status(sim_engine, sim_running)

    with col2:
        _render_live_status(live_engine, live_running)

    with col3:
        _render_system_info(strategy_count, instrument_count)

    st.markdown("---")

    # ============ 快速操作入口 ============
    st.subheader("快速入口")

    col1, col2, col3, col4 = st.columns(4)

    quick_actions = [
        ("模拟交易", "使用真实行情数据进行策略验证", "模拟交易"),
        ("策略回测", "历史数据回测，评估策略表现", "回测系统"),
        ("风控中心", "设置风控规则，监控交易风险", "风控中心"),
        ("系统设置", "配置天勤账号、品种参数等", "系统设置"),
    ]

    for col, (title, desc, nav_page) in zip([col1, col2, col3, col4], quick_actions):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)
            if st.button(f"进入{title}", use_container_width=True, key=f"btn_{nav_page}"):
                st.session_state.nav_page = nav_page
                st.rerun()

    st.markdown("---")

    # ============ 使用说明 ============
    st.subheader("使用流程")
    st.markdown("""
    1. **回测验证** → 在「回测系统」中测试策略，确认参数
    2. **模拟交易** → 在「模拟交易」中使用真实行情验证策略
    3. **实盘上线** → 确认无误后，在「实盘交易」中启动真实交易
    """)


def _render_sim_status(sim_engine, is_running: bool):
    """渲染模拟交易状态"""
    st.subheader("模拟交易")
    if is_running:
        st.success("运行中")
        account = sim_engine.get_account() if hasattr(sim_engine, 'get_account') else None
        if account:
            st.metric("账户权益", f"¥{account.balance:,.0f}")
            positions = sim_engine.get_positions() if hasattr(sim_engine, 'get_positions') else []
            st.metric("持仓数量", f"{len(positions)}")
    else:
        st.info("未启动")
        st.caption("前往「模拟交易」页面启动")


def _render_live_status(live_engine, is_running: bool):
    """渲染实盘交易状态"""
    st.subheader("实盘交易")
    if is_running:
        st.success("运行中")
        account = live_engine.get_account() if hasattr(live_engine, 'get_account') else None
        if account:
            st.metric("账户权益", f"¥{account.balance:,.0f}")
            positions = live_engine.get_positions() if hasattr(live_engine, 'get_positions') else []
            st.metric("持仓数量", f"{len(positions)}")
    else:
        st.warning("未启动")
        st.caption("前往「实盘交易」页面启动")


def _render_system_info(strategy_count: int, instrument_count: int):
    """渲染系统信息"""
    st.subheader("系统信息")
    st.metric("已配置策略", f"{strategy_count}")
    st.metric("已配置品种", f"{instrument_count}")
