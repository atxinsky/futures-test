# coding=utf-8
"""
风控中心页面
风险监控、风控设置、风控日志
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def render_risk_center_page():
    """
    渲染风控中心页面

    显示：
    1. 当前风险状态
    2. 风险指标监控
    3. 风控规则设置
    4. 风控日志
    """
    st.title("风控中心")

    # 获取实时风险数据
    risk_data = _get_risk_data()

    # ============ 风险状态总览 ============
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        risk_level = risk_data.get('level', 'low')
        _render_risk_status(risk_level)

    st.markdown("---")

    # ============ 风险指标 ============
    st.subheader("风险指标")
    _render_risk_metrics(risk_data)

    st.markdown("---")

    # ============ 风控设置 ============
    st.subheader("风控设置")
    _render_risk_settings()

    st.markdown("---")

    # ============ 风控日志 ============
    st.subheader("风控日志")
    _render_risk_logs()


def _get_risk_data() -> Dict[str, Any]:
    """获取风险数据"""
    # 尝试从引擎获取实时数据
    sim_engine = st.session_state.get('sim_engine')
    live_engine = st.session_state.get('live_engine')

    engine = live_engine or sim_engine

    if engine and hasattr(engine, 'get_risk_metrics'):
        try:
            return engine.get_risk_metrics()
        except:
            pass

    # 返回默认/模拟数据
    return {
        'level': 'low',
        'margin_ratio': 0.217,
        'daily_loss': -0.005,
        'max_drawdown': 0.032,
        'consecutive_losses': 1,
        'limits': {
            'margin_ratio': 0.80,
            'daily_loss': 0.05,
            'max_drawdown': 0.15,
            'consecutive_losses': 5,
        }
    }


def _render_risk_status(risk_level: str):
    """渲染风险状态"""
    status_map = {
        'low': ('安全', st.success),
        'medium': ('警告', st.warning),
        'high': ('高风险', st.warning),
        'critical': ('危险', st.error),
    }

    text, func = status_map.get(risk_level, ('未知', st.info))
    func(f"### 风险状态: {text}")


def _render_risk_metrics(risk_data: Dict[str, Any]):
    """渲染风险指标"""
    limits = risk_data.get('limits', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        margin_ratio = risk_data.get('margin_ratio', 0) * 100
        margin_limit = limits.get('margin_ratio', 0.8) * 100
        st.metric("保证金占用", f"{margin_ratio:.1f}%", f"限制: {margin_limit:.0f}%")

    with col2:
        daily_loss = risk_data.get('daily_loss', 0) * 100
        daily_limit = limits.get('daily_loss', 0.05) * 100
        st.metric("日亏损", f"{daily_loss:.1f}%", f"限制: {daily_limit:.0f}%")

    with col3:
        drawdown = risk_data.get('max_drawdown', 0) * 100
        dd_limit = limits.get('max_drawdown', 0.15) * 100
        st.metric("最大回撤", f"{drawdown:.1f}%", f"限制: {dd_limit:.0f}%")

    with col4:
        consec = risk_data.get('consecutive_losses', 0)
        consec_limit = limits.get('consecutive_losses', 5)
        st.metric("连续亏损", f"{consec}次", f"限制: {consec_limit}次")


def _render_risk_settings():
    """渲染风控设置"""
    # 加载当前配置
    from utils.tq_config import load_tq_config
    config = load_tq_config()
    risk_config = config.get('risk_config', {})

    col1, col2 = st.columns(2)

    with col1:
        st.write("**持仓限制**")
        max_pos_per_symbol = st.number_input(
            "单品种最大持仓",
            value=risk_config.get('max_position_per_symbol', 10),
            min_value=1,
            key="risk_max_pos_symbol"
        )
        max_pos_total = st.number_input(
            "总最大持仓",
            value=risk_config.get('max_total_position', 50),
            min_value=1,
            key="risk_max_pos_total"
        )

        st.write("**资金风控**")
        max_margin_ratio = st.slider(
            "最大保证金占用比例",
            0.0, 1.0,
            value=risk_config.get('max_margin_ratio', 0.8),
            key="risk_margin_ratio"
        )
        min_available = st.number_input(
            "最小可用资金",
            value=risk_config.get('min_available', 10000),
            min_value=0,
            key="risk_min_available"
        )

    with col2:
        st.write("**亏损控制**")
        max_daily_loss = st.slider(
            "日最大亏损比例",
            0.0, 0.2,
            value=risk_config.get('max_daily_loss', 0.05),
            key="risk_daily_loss"
        )
        max_drawdown = st.slider(
            "最大回撤比例",
            0.0, 0.3,
            value=risk_config.get('max_drawdown', 0.15),
            key="risk_drawdown"
        )
        max_consecutive = st.number_input(
            "最大连续亏损次数",
            value=risk_config.get('max_consecutive_losses', 5),
            min_value=1,
            key="risk_consecutive"
        )

        st.write("**其他设置**")
        force_close = st.checkbox(
            "达到限制时强制平仓",
            value=risk_config.get('force_close', True),
            key="risk_force_close"
        )
        allow_open = st.checkbox(
            "高风险时允许开仓",
            value=risk_config.get('allow_open_when_high_risk', False),
            key="risk_allow_open"
        )

    if st.button("保存设置", use_container_width=True, key="btn_save_risk"):
        # 保存配置
        new_risk_config = {
            'max_position_per_symbol': max_pos_per_symbol,
            'max_total_position': max_pos_total,
            'max_margin_ratio': max_margin_ratio,
            'min_available': min_available,
            'max_daily_loss': max_daily_loss,
            'max_drawdown': max_drawdown,
            'max_consecutive_losses': max_consecutive,
            'force_close': force_close,
            'allow_open_when_high_risk': allow_open,
        }

        from utils.tq_config import save_tq_config
        config['risk_config'] = new_risk_config
        save_tq_config(config)

        st.success("风控设置已保存!")


def _render_risk_logs():
    """渲染风控日志"""
    # 尝试从审计日志获取
    try:
        from utils.audit_logger import get_audit_logger, AuditEventType

        audit_logger = get_audit_logger()
        risk_events = [
            AuditEventType.RISK_CHECK_PASSED,
            AuditEventType.RISK_CHECK_FAILED,
            AuditEventType.RISK_ALERT,
            AuditEventType.RISK_FORCE_CLOSE,
        ]

        logs = []
        for event_type in risk_events:
            logs.extend(audit_logger.get_recent(count=10, event_type=event_type))

        if logs:
            # 按时间排序
            logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            logs_df = pd.DataFrame([
                {
                    '时间': log.get('timestamp', '')[:19].replace('T', ' '),
                    '级别': log.get('level', 'INFO'),
                    '消息': log.get('message', ''),
                }
                for log in logs[:20]
            ])

            st.dataframe(logs_df, hide_index=True, use_container_width=True)
            return

    except ImportError:
        pass

    # 默认示例数据
    logs_df = pd.DataFrame({
        '时间': [datetime.now().strftime('%H:%M:%S'), '14:20:15', '11:30:00'],
        '级别': ['INFO', 'WARNING', 'INFO'],
        '消息': [
            '订单风控检查通过: RB2505 买开2手',
            '日亏损接近限制: -4.2% (限制: -5%)',
            '新策略加入: WaveTrend'
        ]
    })

    st.dataframe(logs_df, hide_index=True, use_container_width=True)
