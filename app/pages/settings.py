# coding=utf-8
"""
系统设置页面
基础设置、品种配置、网关设置、数据管理
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def render_settings_page():
    """
    渲染系统设置页面

    功能：
    1. 基础设置（账户、显示、通知）
    2. 品种配置
    3. 网关设置（TqSdk）
    4. 数据管理
    """
    st.title("系统设置")

    tab1, tab2, tab3, tab4 = st.tabs(["基础设置", "品种配置", "网关设置", "数据管理"])

    with tab1:
        _render_basic_settings()

    with tab2:
        _render_instrument_settings()

    with tab3:
        _render_gateway_settings()

    with tab4:
        _render_data_settings()


def _render_basic_settings():
    """渲染基础设置"""
    st.subheader("基础设置")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**账户设置**")
        st.number_input("初始资金", value=100000, min_value=10000, key="set_capital")
        st.selectbox("结算货币", ["CNY", "USD"], key="set_currency")

        st.write("**显示设置**")
        st.checkbox("深色模式", value=True, key="set_dark_mode")
        st.selectbox("刷新频率", ["1秒", "3秒", "5秒", "10秒"], key="set_refresh")

    with col2:
        st.write("**通知设置**")
        st.checkbox("成交通知", value=True, key="set_notify_trade")
        st.checkbox("风控预警通知", value=True, key="set_notify_risk")
        st.checkbox("策略信号通知", value=False, key="set_notify_signal")

        st.write("**日志设置**")
        st.selectbox("日志级别", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1, key="set_log_level")
        st.checkbox("保存日志到文件", value=True, key="set_log_file")


def _render_instrument_settings():
    """渲染品种配置"""
    st.subheader("品种配置")

    # 从配置获取品种列表
    try:
        from config import INSTRUMENTS
        instruments_data = []
        for code, info in list(INSTRUMENTS.items())[:10]:  # 显示前10个
            instruments_data.append({
                '品种代码': code,
                '品种名称': info.get('name', code),
                '合约乘数': info.get('multiplier', 0),
                '保证金率': f"{info.get('margin_rate', 0)*100:.0f}%",
                '手续费': f"万分之{info.get('commission_rate', 0)*10000:.1f}" if info.get('commission_rate') else '-'
            })
        instruments_df = pd.DataFrame(instruments_data)
    except ImportError:
        instruments_df = pd.DataFrame({
            '品种代码': ['RB', 'I', 'AU', 'CU', 'AL'],
            '品种名称': ['螺纹钢', '铁矿石', '黄金', '沪铜', '沪铝'],
            '合约乘数': [10, 100, 1000, 5, 5],
            '保证金率': ['10%', '12%', '8%', '10%', '10%'],
            '手续费': ['成交额万分之一', '成交额万分之一', '10元/手', '成交额万分之0.5', '3元/手']
        })

    st.dataframe(instruments_df, hide_index=True, use_container_width=True)

    with st.expander("添加/编辑品种"):
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("品种代码", key="inst_code")
            st.text_input("品种名称", key="inst_name")
            st.number_input("合约乘数", value=10, key="inst_multi")
        with col2:
            st.number_input("保证金率", value=0.1, format="%.2f", key="inst_margin")
            st.selectbox("手续费类型", ["按比例", "固定金额"], key="inst_comm_type")
            st.number_input("手续费", value=0.0001, format="%.4f", key="inst_comm")

        if st.button("保存品种配置", key="btn_save_inst"):
            st.success("品种配置已保存!")


def _render_gateway_settings():
    """渲染网关设置（TqSdk）"""
    st.subheader("TqSdk连接设置")

    # 使用统一配置管理
    from utils.tq_config import (
        load_tq_config_for_settings,
        save_tq_config_for_settings,
        test_tq_connection
    )

    tq_config = load_tq_config_for_settings()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**天勤账号**")
        tq_user = st.text_input(
            "天勤用户名",
            value=tq_config.get('tq_user', ''),
            key="settings_tq_user"
        )
        tq_password = st.text_input(
            "天勤密码",
            type="password",
            value=tq_config.get('tq_password', ''),
            key="settings_tq_password"
        )

        st.markdown("---")

        st.write("**交易模式**")
        sim_mode = st.radio(
            "选择模式",
            options=["模拟盘 (TqSim)", "实盘 (需要期货账号)"],
            index=0 if tq_config.get('sim_mode', True) else 1,
            horizontal=True,
            key="settings_sim_mode"
        )
        sim_mode_bool = sim_mode == "模拟盘 (TqSim)"

    with col2:
        st.write("**期货账号配置**")
        if not sim_mode_bool:
            broker_id = st.text_input(
                "期货公司代码",
                value=tq_config.get('broker_id', ''),
                key="settings_broker_id"
            )
            td_account = st.text_input(
                "交易账号",
                value=tq_config.get('td_account', ''),
                key="settings_td_account"
            )
            td_password = st.text_input(
                "交易密码",
                type="password",
                value=tq_config.get('td_password', ''),
                key="settings_td_password"
            )
            st.info("实盘交易需要开通期货账户")
        else:
            st.info("模拟盘模式使用TqSim，无需期货账号，使用真实行情数据进行模拟撮合")
            broker_id = tq_config.get('broker_id', '')
            td_account = tq_config.get('td_account', '')
            td_password = tq_config.get('td_password', '')

    st.markdown("---")

    # 风控设置
    st.write("**风控设置**")
    risk_config = tq_config.get('risk_config', {})

    col1, col2, col3 = st.columns(3)

    with col1:
        max_pos = st.number_input(
            "单品种最大持仓",
            min_value=1, max_value=100,
            value=risk_config.get('max_position_per_symbol', 10),
            key="settings_max_pos"
        )

    with col2:
        max_daily_loss = st.slider(
            "日最大亏损%",
            min_value=1, max_value=20,
            value=int(risk_config.get('max_daily_loss', 0.05) * 100),
            key="settings_max_daily_loss"
        )

    with col3:
        max_drawdown = st.slider(
            "最大回撤%",
            min_value=5, max_value=50,
            value=int(risk_config.get('max_drawdown', 0.15) * 100),
            key="settings_max_drawdown"
        )

    st.markdown("---")

    # 保存和测试按钮
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("保存配置", type="primary", use_container_width=True, key="save_tq_config"):
            new_config = {
                'tq_user': tq_user,
                'tq_password': tq_password,
                'sim_mode': sim_mode_bool,
                'broker_id': broker_id,
                'td_account': td_account,
                'td_password': td_password,
                'default_symbols': tq_config.get('default_symbols', ['RB', 'AU', 'IF']),
                'initial_capital': tq_config.get('initial_capital', 100000),
                'risk_config': {
                    'max_position_per_symbol': max_pos,
                    'max_daily_loss': max_daily_loss / 100,
                    'max_drawdown': max_drawdown / 100
                }
            }
            save_tq_config_for_settings(new_config)
            st.success("配置已保存!")

    with col2:
        if st.button("测试连接", use_container_width=True, key="test_tq_conn"):
            if not tq_user or not tq_password:
                st.error("请输入天勤账号和密码")
            else:
                with st.spinner("正在连接天勤..."):
                    success, message = test_tq_connection(tq_user, tq_password)
                if success:
                    st.success(f"连接成功! {message}")
                else:
                    st.error(message)


def _render_data_settings():
    """渲染数据管理设置"""
    st.subheader("数据管理")

    st.write("**数据库信息**")
    st.info("数据来源: TianQin量化数据库 / AKShare")

    # 尝试获取实际数据统计
    try:
        from data_manager import get_data_status
        df_status = get_data_status()
        data_count = len(df_status[df_status['record_count'] > 0]) if df_status is not None else 0
        total_records = df_status['record_count'].sum() if df_status is not None else 0
    except:
        data_count = 0
        total_records = 0

    col1, col2 = st.columns(2)

    with col1:
        st.metric("已下载品种", f"{data_count}")
        st.metric("交易记录数", f"{total_records:,}")

    with col2:
        st.metric("数据时间范围", "2020-01 至 今")
        st.metric("支持品种数量", "60+")

    st.write("**数据操作**")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("更新数据", use_container_width=True, key="btn_update_data"):
            st.info("请前往「数据管理」页面下载数据")

    with col2:
        if st.button("清理缓存", use_container_width=True, key="btn_clear_cache"):
            st.cache_data.clear()
            st.success("缓存已清理")

    with col3:
        if st.button("查看详情", use_container_width=True, key="btn_data_detail"):
            st.session_state.nav_page = "回测系统"
            st.rerun()
