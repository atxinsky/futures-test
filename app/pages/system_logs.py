# coding=utf-8
"""
系统日志页面
显示系统运行日志、回测日志、交易日志
"""

import streamlit as st
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from collections import deque
import threading
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class LogBuffer:
    """
    日志缓冲区 - 用于捕获实时日志
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._logs = deque(maxlen=1000)
                    cls._instance._handler = None
        return cls._instance

    def get_logs(self, limit: int = 100) -> List[dict]:
        """获取最近的日志"""
        return list(self._logs)[-limit:]

    def add_log(self, level: str, message: str, module: str = ""):
        """添加日志"""
        self._logs.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': level,
            'module': module,
            'message': message
        })

    def clear(self):
        """清空日志"""
        self._logs.clear()


class StreamlitLogHandler(logging.Handler):
    """
    Streamlit日志处理器 - 将日志发送到LogBuffer
    """
    def __init__(self, buffer: LogBuffer):
        super().__init__()
        self.buffer = buffer
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record):
        try:
            msg = self.format(record)
            self.buffer.add_log(
                level=record.levelname,
                message=msg,
                module=record.name
            )
        except:
            pass


def setup_log_capture():
    """设置日志捕获"""
    buffer = LogBuffer()

    # 检查是否已经设置了handler
    if buffer._handler is None:
        handler = StreamlitLogHandler(buffer)
        handler.setLevel(logging.INFO)

        # 添加到root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        buffer._handler = handler

    return buffer


def render_system_logs_page():
    """渲染系统日志页面"""
    st.markdown("## 📋 系统日志")

    # 设置日志捕获
    log_buffer = setup_log_capture()

    # 顶部控制栏
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
        log_source = st.selectbox(
            "日志来源",
            ["实时日志", "回测日志", "交易日志", "系统日志"],
            key="log_source"
        )

    with col2:
        log_level = st.selectbox(
            "日志级别",
            ["全部", "DEBUG", "INFO", "WARNING", "ERROR"],
            key="log_level"
        )

    with col3:
        auto_refresh = st.checkbox("自动刷新", value=False, key="log_auto_refresh")

    with col4:
        if st.button("刷新", key="refresh_logs"):
            st.rerun()

    with col5:
        if st.button("清空日志", key="clear_logs"):
            log_buffer.clear()
            st.success("日志已清空")
            st.rerun()

    st.markdown("---")

    if log_source == "实时日志":
        _render_realtime_logs(log_buffer, log_level)
    elif log_source == "回测日志":
        _render_backtest_logs(log_level)
    elif log_source == "交易日志":
        _render_trading_logs(log_level)
    else:
        _render_system_file_logs(log_level)

    # 自动刷新
    if auto_refresh:
        _auto_refresh(3)


def _render_realtime_logs(buffer: LogBuffer, level_filter: str):
    """渲染实时日志"""
    st.markdown("### 实时日志")

    logs = buffer.get_logs(200)

    if not logs:
        st.info("暂无日志")

        # 添加测试日志按钮
        if st.button("生成测试日志"):
            logger = logging.getLogger("test")
            logger.info("这是一条测试INFO日志")
            logger.warning("这是一条测试WARNING日志")
            logger.error("这是一条测试ERROR日志")
            st.rerun()
        return

    # 筛选
    if level_filter != "全部":
        logs = [l for l in logs if l['level'] == level_filter]

    # 显示
    _display_logs(logs)


def _render_backtest_logs(level_filter: str):
    """渲染回测日志"""
    st.markdown("### 回测日志")

    # 从数据库获取回测记录
    try:
        from utils.backtest_storage import get_backtest_storage
        storage = get_backtest_storage()

        records = storage.get_records(limit=20)

        if not records:
            st.info("暂无回测记录")
            return

        # 构建日志
        logs = []
        for r in records:
            level = "INFO"
            if r.total_return < 0:
                level = "WARNING"
            if r.max_drawdown > 0.2:
                level = "ERROR"

            logs.append({
                'timestamp': r.created_at,
                'level': level,
                'module': f"backtest.{r.backtest_type}",
                'message': f"[{r.strategy_name}] {r.symbols[:30]}... | "
                          f"收益:{r.total_return*100:+.1f}% | 回撤:{r.max_drawdown*100:.1f}% | "
                          f"夏普:{r.sharpe_ratio:.2f} | 交易:{r.total_trades}笔"
            })

        # 筛选
        if level_filter != "全部":
            logs = [l for l in logs if l['level'] == level_filter]

        _display_logs(logs)

    except ImportError:
        st.warning("回测存储模块未加载")


def _render_trading_logs(level_filter: str):
    """渲染交易日志"""
    st.markdown("### 交易日志")

    # 检查是否有活跃的交易引擎
    sim_engine = st.session_state.get('sim_engine')
    live_engine = st.session_state.get('live_engine')

    logs = []

    # 从交易引擎获取日志
    if sim_engine:
        # 获取最近的订单/成交记录作为日志
        try:
            orders = getattr(sim_engine, '_recent_orders', [])
            for o in orders[-50:]:
                logs.append({
                    'timestamp': str(o.get('time', '')),
                    'level': 'INFO',
                    'module': 'sim_trading',
                    'message': f"[模拟] {o.get('symbol', '')} {o.get('direction', '')} "
                              f"{o.get('volume', 0)}手 @ {o.get('price', 0):.2f}"
                })
        except:
            pass

    if live_engine:
        try:
            orders = getattr(live_engine, '_recent_orders', [])
            for o in orders[-50:]:
                logs.append({
                    'timestamp': str(o.get('time', '')),
                    'level': 'INFO',
                    'module': 'live_trading',
                    'message': f"[实盘] {o.get('symbol', '')} {o.get('direction', '')} "
                              f"{o.get('volume', 0)}手 @ {o.get('price', 0):.2f}"
                })
        except:
            pass

    if not logs:
        st.info("暂无交易日志。启动模拟交易或实盘交易后会在此显示。")
        return

    # 筛选
    if level_filter != "全部":
        logs = [l for l in logs if l['level'] == level_filter]

    _display_logs(logs)


def _render_system_file_logs(level_filter: str):
    """渲染系统文件日志"""
    st.markdown("### 系统日志文件")

    # 查找日志文件
    log_dir = Path(__file__).parent.parent.parent / "logs"
    data_dir = Path(__file__).parent.parent.parent / "data"

    log_files = []

    if log_dir.exists():
        log_files.extend(list(log_dir.glob("*.log")))

    if data_dir.exists():
        log_files.extend(list(data_dir.glob("*.log")))

    if not log_files:
        st.info("未找到日志文件")

        # 显示预期的日志目录
        st.write(f"日志目录: `{log_dir}`")
        st.write(f"数据目录: `{data_dir}`")
        return

    # 选择日志文件
    selected_file = st.selectbox(
        "选择日志文件",
        log_files,
        format_func=lambda x: x.name
    )

    if selected_file:
        # 读取日志文件
        try:
            with open(selected_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[-500:]  # 最后500行

            # 解析日志
            logs = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 尝试解析日志格式
                level = "INFO"
                if " ERROR " in line or "ERROR:" in line:
                    level = "ERROR"
                elif " WARNING " in line or "WARNING:" in line:
                    level = "WARNING"
                elif " DEBUG " in line or "DEBUG:" in line:
                    level = "DEBUG"

                logs.append({
                    'timestamp': '',
                    'level': level,
                    'module': '',
                    'message': line
                })

            # 筛选
            if level_filter != "全部":
                logs = [l for l in logs if l['level'] == level_filter]

            _display_logs(logs)

        except Exception as e:
            st.error(f"读取日志失败: {e}")


def _display_logs(logs: List[dict]):
    """显示日志列表"""
    if not logs:
        st.info("没有符合条件的日志")
        return

    # 日志样式
    level_colors = {
        'DEBUG': '#6c757d',
        'INFO': '#17a2b8',
        'WARNING': '#ffc107',
        'ERROR': '#dc3545'
    }

    # 显示日志数量
    st.caption(f"共 {len(logs)} 条日志")

    # 搜索框
    search = st.text_input("搜索日志", key="log_search", placeholder="输入关键词筛选...")

    if search:
        logs = [l for l in logs if search.lower() in l['message'].lower()]

    # 构建HTML
    log_html = """
    <style>
    .log-container { font-family: 'Consolas', monospace; font-size: 12px; }
    .log-line { padding: 4px 8px; border-bottom: 1px solid #eee; }
    .log-line:hover { background: #f8f9fa; }
    .log-time { color: #6c757d; margin-right: 8px; }
    .log-level { padding: 1px 6px; border-radius: 3px; margin-right: 8px; font-size: 10px; }
    .log-module { color: #6c757d; margin-right: 8px; }
    </style>
    <div class="log-container">
    """

    for log in reversed(logs[-100:]):  # 最新的在上面
        color = level_colors.get(log['level'], '#6c757d')
        timestamp = log['timestamp'] if log['timestamp'] else ''
        module = f"[{log['module']}]" if log['module'] else ''

        log_html += f"""
        <div class="log-line">
            <span class="log-time">{timestamp}</span>
            <span class="log-level" style="background: {color}; color: white;">{log['level']}</span>
            <span class="log-module">{module}</span>
            <span class="log-message">{log['message']}</span>
        </div>
        """

    log_html += "</div>"

    st.markdown(log_html, unsafe_allow_html=True)

    # 下载日志
    st.markdown("---")
    if st.button("下载日志"):
        log_text = "\n".join([
            f"{l['timestamp']} [{l['level']}] {l['module']} {l['message']}"
            for l in logs
        ])
        st.download_button(
            "下载",
            log_text.encode('utf-8'),
            f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain"
        )


def _auto_refresh(interval: int):
    """自动刷新"""
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=interval * 1000, key="log_auto_refresh_timer")
    except ImportError:
        st.markdown(
            f"""
            <script>
                setTimeout(function() {{
                    window.location.reload();
                }}, {interval * 1000});
            </script>
            """,
            unsafe_allow_html=True
        )
