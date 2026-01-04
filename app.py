# coding=utf-8
"""
期货策略回测系统 V2.0
专业级界面 - 支持YAML配置文件
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys
import yaml

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import INSTRUMENTS, get_instrument, EXCHANGES
from engine import run_backtest, run_backtest_with_strategy, calculate_indicators
from data_manager import (
    get_data_status, download_symbol, download_batch, load_from_database,
    get_symbol_list_by_category, FUTURES_SYMBOLS, export_to_csv,
    MINUTE_PERIODS, download_minute_symbol, load_minute_from_database, get_minute_data_status
)
from strategies import (
    get_all_strategies, get_strategy, list_strategies,
    load_strategy_from_file, BaseStrategy, StrategyParam
)
from config_manager import (
    list_configs, load_config, save_config, delete_config,
    config_to_yaml, yaml_to_config, create_default_config,
    get_strategy_param_groups, STRATEGY_DEFAULTS, DEFAULT_CONFIG
)

st.set_page_config(
    page_title="期货策略回测系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 专业级CSS样式 ====================
st.markdown("""
<style>
    /* 主题色彩 */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --bg-dark: #1e1e2e;
        --bg-card: #262637;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --border: #374151;
    }

    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* 顶部导航栏 */
    .top-header {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        padding: 1rem 2rem;
        border-radius: 0 0 12px 12px;
        margin: -1rem -1rem 1.5rem -1rem;
        border-bottom: 1px solid #374151;
    }

    .top-header h1 {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .top-header .version {
        background: #4f46e5;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* 配置卡片 */
    .config-card {
        background: linear-gradient(145deg, #262637 0%, #1e1e2e 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .config-card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #374151;
    }

    .config-card-header h3 {
        color: #f1f5f9;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0;
    }

    .config-card-header .icon {
        font-size: 1.1rem;
    }

    /* YAML编辑器样式 */
    .yaml-editor {
        background: #1a1a2e;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #e2e8f0;
    }

    .yaml-key { color: #7dd3fc; }
    .yaml-value { color: #fbbf24; }
    .yaml-comment { color: #6b7280; font-style: italic; }

    /* 指标卡片 */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .metric-box {
        background: linear-gradient(145deg, #262637 0%, #1e1e2e 100%);
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .metric-box.profit {
        border-color: #10b981;
        background: linear-gradient(145deg, #0f3d2e 0%, #1e1e2e 100%);
    }

    .metric-box.loss {
        border-color: #ef4444;
        background: linear-gradient(145deg, #3d1f1f 0%, #1e1e2e 100%);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        color: #f1f5f9;
        font-size: 1.25rem;
        font-weight: 700;
    }

    .metric-value.green { color: #10b981; }
    .metric-value.red { color: #ef4444; }

    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }

    /* 策略选择器 */
    .strategy-selector {
        background: #1e1e2e;
        border: 2px solid #4f46e5;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .strategy-name {
        color: #f1f5f9;
        font-size: 1.1rem;
        font-weight: 600;
    }

    .strategy-version {
        color: #6366f1;
        font-size: 0.8rem;
    }

    /* 参数分组 */
    .param-group {
        background: #1a1a2e;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
    }

    .param-group-title {
        color: #a5b4fc;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }

    /* 运行按钮特殊样式 */
    .run-btn > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        width: 100%;
        padding: 0.8rem !important;
        font-size: 1rem !important;
    }

    .run-btn > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4) !important;
    }

    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 0.25rem;
        gap: 0.25rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8;
        padding: 0.5rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: #4f46e5 !important;
        color: white !important;
    }

    /* 表格样式 */
    .dataframe {
        background: #1e1e2e !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }

    .dataframe th {
        background: #262637 !important;
        color: #e2e8f0 !important;
        border-bottom: 1px solid #374151 !important;
    }

    .dataframe td {
        color: #cbd5e1 !important;
        border-bottom: 1px solid #2d2d44 !important;
    }

    /* 代码块样式 */
    .stCodeBlock {
        background: #0d0d14 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }

    /* 侧边栏样式 */
    .css-1d391kg {
        background: #1e1e2e;
    }

    /* 输入框样式 */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: #1a1a2e !important;
        border: 1px solid #374151 !important;
        color: #e2e8f0 !important;
        border-radius: 6px !important;
    }

    /* 滑块样式 */
    .stSlider > div > div > div > div {
        background: #4f46e5 !important;
    }

    /* Expander样式 */
    .streamlit-expanderHeader {
        background: #262637 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }

    /* 告警框样式 */
    .stAlert {
        background: #262637;
        border: 1px solid #374151;
        border-radius: 8px;
    }

    /* 文件状态徽章 */
    .file-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
    }

    .file-badge.saved { background: #10b981; color: white; }
    .file-badge.modified { background: #f59e0b; color: white; }
    .file-badge.new { background: #6366f1; color: white; }

    /* 交易结果颜色 */
    .trade-win { color: #10b981 !important; font-weight: 600; }
    .trade-loss { color: #ef4444 !important; font-weight: 600; }

    /* 响应式布局 */
    @media (max-width: 768px) {
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """渲染顶部标题"""
    st.markdown("""
    <div class="top-header">
        <h1>📊 期货策略回测系统 <span class="version">v2.0</span></h1>
    </div>
    """, unsafe_allow_html=True)


def resample_data(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """重采样数据到不同周期"""
    if period == "日线":
        return df

    df = df.copy()
    df = df.set_index('time')

    if period == "周线":
        rule = 'W'
    elif period == "月线":
        rule = 'ME'
    else:
        return df.reset_index()

    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum' if 'volume' in df.columns else 'first'
    }).dropna()

    return resampled.reset_index()


@st.cache_data
def load_data_from_db(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """从数据库加载数据"""
    df = load_from_database(symbol, start_date, end_date)
    return df


def render_config_editor():
    """渲染配置编辑器"""
    st.markdown("""
    <div class="config-card">
        <div class="config-card-header">
            <span class="icon">⚙️</span>
            <h3>配置文件</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 配置文件操作
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        configs = list_configs()
        config_options = ["新建配置..."] + configs

        if 'current_config' not in st.session_state:
            st.session_state.current_config = "新建配置..."

        selected_config = st.selectbox(
            "📁 选择配置文件",
            options=config_options,
            index=config_options.index(st.session_state.current_config) if st.session_state.current_config in config_options else 0,
            key="config_selector"
        )

    with col2:
        if st.button("💾 保存", use_container_width=True):
            if 'config_yaml' in st.session_state:
                try:
                    config = yaml_to_config(st.session_state.config_yaml)
                    filename = st.session_state.get('config_filename', 'untitled.yml')
                    save_config(filename, config)
                    st.success(f"已保存: {filename}")
                    st.rerun()
                except Exception as e:
                    st.error(f"保存失败: {e}")

    with col3:
        if selected_config != "新建配置..." and st.button("🗑️ 删除", use_container_width=True):
            delete_config(selected_config)
            st.session_state.current_config = "新建配置..."
            st.rerun()

    # 加载或创建配置
    if selected_config == "新建配置...":
        # 新建配置
        col1, col2 = st.columns(2)
        with col1:
            new_filename = st.text_input("配置文件名", value="backtest_config", key="new_config_name")
            st.session_state.config_filename = new_filename + ".yml"
        with col2:
            strategies = get_all_strategies()
            selected_strategy = st.selectbox(
                "选择策略模板",
                options=list(strategies.keys()),
                format_func=lambda x: strategies[x].display_name,
                key="new_strategy_select"
            )

        if 'config_yaml' not in st.session_state or st.session_state.get('last_strategy') != selected_strategy:
            config = create_default_config(selected_strategy)
            st.session_state.config_yaml = config_to_yaml(config)
            st.session_state.last_strategy = selected_strategy
    else:
        # 加载已有配置
        st.session_state.current_config = selected_config
        st.session_state.config_filename = selected_config
        if 'config_yaml' not in st.session_state or st.session_state.get('loaded_config') != selected_config:
            config = load_config(selected_config)
            st.session_state.config_yaml = config_to_yaml(config)
            st.session_state.loaded_config = selected_config

    return st.session_state.get('config_yaml', '')


def render_yaml_editor(yaml_content: str) -> str:
    """渲染YAML编辑器"""
    st.markdown("""
    <div class="config-card">
        <div class="config-card-header">
            <span class="icon">📝</span>
            <h3>YAML 配置</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # YAML编辑器
    edited_yaml = st.text_area(
        "编辑配置 (YAML格式)",
        value=yaml_content,
        height=400,
        key="yaml_editor",
        label_visibility="collapsed"
    )

    st.session_state.config_yaml = edited_yaml

    # 解析并验证
    try:
        config = yaml_to_config(edited_yaml)
        st.success("✓ YAML语法正确")
        return config
    except Exception as e:
        st.error(f"✗ YAML语法错误: {e}")
        return None


def render_visual_config(config: dict):
    """渲染可视化配置面板"""
    if config is None:
        return None

    st.markdown("""
    <div class="config-card">
        <div class="config-card-header">
            <span class="icon">🎛️</span>
            <h3>可视化配置</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 基础配置
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📌 基础设置**")
        config['initial_capital'] = st.number_input(
            "初始资金",
            value=config.get('initial_capital', 1000000),
            min_value=100000,
            step=100000,
            format="%d"
        )

    with col2:
        st.markdown("**📅 回测时间**")
        # 解析日期
        try:
            start_str = str(config.get('time_start', '20200101'))
            end_str = str(config.get('time_end', '20251231'))
            start_date = datetime.strptime(start_str, '%Y%m%d').date()
            end_date = datetime.strptime(end_str, '%Y%m%d').date()
        except:
            start_date = datetime(2020, 1, 1).date()
            end_date = datetime(2025, 12, 31).date()

        new_start = st.date_input("起始日期", value=start_date, key="vis_start_date")
        config['time_start'] = new_start.strftime('%Y%m%d')

    with col3:
        st.markdown("**⏱️ 周期**")
        new_end = st.date_input("结束日期", value=end_date, key="vis_end_date")
        config['time_end'] = new_end.strftime('%Y%m%d')

        timeframe_options = ["日线", "周线", "月线", "60分钟", "30分钟", "15分钟"]
        current_tf = config.get('run_policy', {}).get('timeframes', '日线')
        if current_tf not in timeframe_options:
            current_tf = "日线"
        config['run_policy']['timeframes'] = st.selectbox(
            "K线周期",
            options=timeframe_options,
            index=timeframe_options.index(current_tf),
            key="vis_timeframe"
        )

    st.markdown("---")

    # 品种选择
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**📈 交易品种**")
        df_status = get_data_status()
        symbols_with_data = df_status[df_status['record_count'] > 0]['symbol'].tolist()

        current_pairs = config.get('pairs', ['IF'])
        if isinstance(current_pairs, list) and len(current_pairs) > 0:
            current_symbol = current_pairs[0]
        else:
            current_symbol = 'IF'

        if current_symbol not in symbols_with_data and symbols_with_data:
            current_symbol = symbols_with_data[0]

        selected_symbol = st.selectbox(
            "选择品种",
            options=symbols_with_data if symbols_with_data else ['IF'],
            index=symbols_with_data.index(current_symbol) if current_symbol in symbols_with_data else 0,
            format_func=lambda x: f"{x} - {FUTURES_SYMBOLS.get(x, ('未知',))[0]}",
            key="vis_symbol"
        )
        config['pairs'] = [selected_symbol]

    with col2:
        st.markdown("**🎯 策略参数**")

        strategy_name = config.get('run_policy', {}).get('name', 'brother2v6')
        params = config.get('run_policy', {}).get('params', {})

        # 获取参数分组
        param_groups = get_strategy_param_groups(strategy_name)

        # 获取策略类以获取参数定义
        strategies = get_all_strategies()
        strategy_class = strategies.get(strategy_name)

        if strategy_class:
            param_defs = {p.name: p for p in strategy_class.get_params()}

            # 按分组显示参数
            for group_name, param_names in param_groups.items():
                with st.expander(f"📦 {group_name}", expanded=True):
                    cols = st.columns(3)
                    for i, param_name in enumerate(param_names):
                        if param_name in param_defs:
                            p = param_defs[param_name]
                            with cols[i % 3]:
                                if p.param_type == 'int':
                                    params[param_name] = st.number_input(
                                        p.label,
                                        value=int(params.get(param_name, p.default)),
                                        min_value=int(p.min_val) if p.min_val else 1,
                                        max_value=int(p.max_val) if p.max_val else 100,
                                        step=int(p.step) if p.step else 1,
                                        key=f"vis_{param_name}"
                                    )
                                elif p.param_type == 'float':
                                    params[param_name] = st.number_input(
                                        p.label,
                                        value=float(params.get(param_name, p.default)),
                                        min_value=float(p.min_val) if p.min_val else 0.0,
                                        max_value=float(p.max_val) if p.max_val else 100.0,
                                        step=float(p.step) if p.step else 0.1,
                                        format="%.2f",
                                        key=f"vis_{param_name}"
                                    )

            config['run_policy']['params'] = params

    # 更新YAML
    st.session_state.config_yaml = config_to_yaml(config)

    return config


def run_backtest_from_config(config: dict):
    """根据配置运行回测"""
    if config is None:
        st.error("配置无效，请检查YAML格式")
        return None

    try:
        # 解析配置
        strategy_name = config.get('run_policy', {}).get('name', 'brother2v6')
        params = config.get('run_policy', {}).get('params', {})
        timeframe = config.get('run_policy', {}).get('timeframes', '日线')
        symbol = config.get('pairs', ['IF'])[0] if config.get('pairs') else 'IF'
        initial_capital = config.get('initial_capital', 1000000)

        start_str = str(config.get('time_start', '20200101'))
        end_str = str(config.get('time_end', '20251231'))

        # 转换日期格式
        start_date = f"{start_str[:4]}-{start_str[4:6]}-{start_str[6:8]}"
        end_date = f"{end_str[:4]}-{end_str[4:6]}-{end_str[6:8]}"

        # 加载数据
        if timeframe in ["5分钟", "15分钟", "30分钟", "60分钟"]:
            period_map = {"5分钟": "5", "15分钟": "15", "30分钟": "30", "60分钟": "60"}
            period = period_map[timeframe]
            df_data = load_minute_from_database(symbol, period, start_date, end_date)
            if len(df_data) == 0:
                st.error(f"没有 {timeframe} 数据，请先下载分钟数据")
                return None
        else:
            df_data = load_from_database(symbol, start_date, end_date)
            if len(df_data) == 0:
                st.error("没有数据，请先下载数据")
                return None
            df_data = resample_data(df_data, timeframe)

        st.info(f"📊 数据: {len(df_data)} 条 ({start_date} ~ {end_date}) - {timeframe}")

        # 获取策略类并创建实例
        strategies = get_all_strategies()
        strategy_class = strategies.get(strategy_name)
        if not strategy_class:
            st.error(f"未找到策略: {strategy_name}")
            return None

        strategy_instance = strategy_class(params)

        # 运行回测
        result = run_backtest_with_strategy(df_data, symbol, strategy_instance, initial_capital)

        return result, df_data

    except Exception as e:
        st.error(f"回测失败: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def render_metrics(result):
    """渲染指标面板"""
    st.markdown("""
    <div class="config-card">
        <div class="config-card-header">
            <span class="icon">📊</span>
            <h3>回测结果</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 主要指标
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    profit_class = "profit" if result.total_pnl > 0 else "loss"
    profit_color = "green" if result.total_pnl > 0 else "red"

    with col1:
        st.metric("总收益", f"¥{result.total_pnl:,.0f}", f"{result.total_return_pct:+.2f}%")
    with col2:
        st.metric("年化收益", f"{result.annual_return_pct:.2f}%")
    with col3:
        st.metric("最大回撤", f"{result.max_drawdown_pct:.2f}%")
    with col4:
        st.metric("夏普比率", f"{result.sharpe_ratio:.2f}")
    with col5:
        win_count = len([t for t in result.trades if t.pnl > 0])
        st.metric("胜率", f"{result.win_rate:.1f}%", f"{win_count}/{len(result.trades)}")
    with col6:
        st.metric("盈亏比", f"{result.profit_factor:.2f}")

    st.markdown("---")

    # 详细指标
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**💰 收益指标**")
        st.write(f"初始资金: ¥{result.initial_capital:,.0f}")
        st.write(f"期末资金: ¥{result.final_capital:,.0f}")
        st.write(f"总盈亏: ¥{result.total_pnl:,.0f}")
        st.write(f"总收益率: {result.total_return_pct:.2f}%")
        st.write(f"年化收益: {result.annual_return_pct:.2f}%")
        st.write(f"总手续费: ¥{result.total_commission:,.0f}")

    with col2:
        st.markdown("**📉 风险指标**")
        st.write(f"最大回撤: {result.max_drawdown_pct:.2f}%")
        st.write(f"回撤金额: ¥{result.max_drawdown_val:,.0f}")
        st.write(f"夏普比率: {result.sharpe_ratio:.2f}")
        st.write(f"索提诺比率: {result.sortino_ratio:.2f}")
        st.write(f"卡尔玛比率: {result.calmar_ratio:.2f}")

    with col3:
        st.markdown("**📈 交易指标**")
        st.write(f"总交易数: {len(result.trades)}")
        st.write(f"胜率: {result.win_rate:.1f}%")
        st.write(f"盈亏比: {result.profit_factor:.2f}")
        st.write(f"平均盈利: ¥{result.avg_win:,.0f}")
        st.write(f"平均亏损: ¥{result.avg_loss:,.0f}")
        st.write(f"平均持仓: {result.avg_holding_days:.1f}天")


def render_equity_chart(result):
    """渲染资金曲线"""
    df = result.equity_curve

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('账户净值', '回撤')
    )

    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['equity'],
            name='账户净值',
            line=dict(color='#6366f1', width=2),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.1)'
        ),
        row=1, col=1
    )

    for trade in result.trades:
        color = '#10b981' if trade.pnl > 0 else '#ef4444'
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_time],
                y=[trade.capital_before if trade.capital_before > 0 else result.initial_capital],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='#6366f1'),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[trade.exit_time],
                y=[trade.capital_after],
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color=color),
                showlegend=False
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=-df['drawdown_pct'],
            name='回撤',
            line=dict(color='#ef4444', width=1),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.3)'
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=500,
        hovermode='x unified',
        showlegend=True,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,46,1)'
    )

    st.plotly_chart(fig, use_container_width=True)


def render_trades_table(result):
    """渲染交易记录"""
    if not result.trades:
        st.warning("没有交易记录")
        return

    trades_data = []
    for t in result.trades:
        trades_data.append({
            '编号': t.trade_id + 1,
            '入场时间': t.entry_time.strftime('%Y-%m-%d'),
            '出场时间': t.exit_time.strftime('%Y-%m-%d') if t.exit_time else '',
            '方向': '多' if t.direction == 1 else '空',
            '入场价': f"{t.entry_price:.2f}",
            '出场价': f"{t.exit_price:.2f}" if t.exit_price else '',
            '手数': t.volume,
            '持仓(天)': t.holding_days,
            '盈亏%': f"{t.pnl_pct:+.2f}%",
            '盈亏额': f"¥{t.pnl:+,.0f}",
            '出场原因': t.exit_tag,
            '结果': '盈' if t.pnl > 0 else '亏'
        })

    df_trades = pd.DataFrame(trades_data)

    col1, col2 = st.columns(2)
    with col1:
        result_filter = st.multiselect("筛选结果", options=['盈', '亏'], default=['盈', '亏'])
    with col2:
        exit_tags = df_trades['出场原因'].unique().tolist()
        tag_filter = st.multiselect("筛选出场原因", options=exit_tags, default=exit_tags)

    df_filtered = df_trades[
        (df_trades['结果'].isin(result_filter)) &
        (df_trades['出场原因'].isin(tag_filter))
    ]

    st.dataframe(df_filtered, use_container_width=True, hide_index=True)

    csv = df_filtered.to_csv(index=False, encoding='utf-8-sig')
    st.download_button("📥 下载交易记录", csv, "trades.csv", "text/csv")


def render_statistics(result):
    """渲染统计分析"""
    if not result.trades:
        st.warning("没有交易记录")
        return

    col1, col2 = st.columns(2)

    with col1:
        if result.exit_tag_stats is not None:
            st.markdown("**出场原因统计**")
            df_exit = result.exit_tag_stats.reset_index()
            df_exit.columns = ['出场原因', '次数', '总盈亏', '平均盈亏', '平均收益%']
            st.dataframe(df_exit, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**收益分布**")
        pnl_list = [t.pnl for t in result.trades]

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=pnl_list, nbinsx=20, marker_color='#6366f1'))
        fig.add_vline(x=0, line_dash="dash", line_color="#ef4444")
        fig.update_layout(
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,46,1)'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_data_management():
    """渲染数据管理页面"""
    st.header("📥 数据管理")

    tab1, tab2, tab3 = st.tabs(["日线数据", "分钟数据", "数据状态"])

    with tab1:
        st.subheader("下载期货数据")
        categories = get_symbol_list_by_category()

        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("选择类别", options=list(categories.keys()))
            symbols_in_cat = categories[category]
            selected_symbols = st.multiselect(
                "选择品种",
                options=[s[0] for s in symbols_in_cat],
                format_func=lambda x: f"{x} - {FUTURES_SYMBOLS[x][0]}",
                default=[s[0] for s in symbols_in_cat[:2]] if symbols_in_cat else []
            )

        with col2:
            st.write("**快捷选择:**")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("股指期货"):
                    st.session_state.quick_select = ["IF", "IH", "IC", "IM"]
            with col_b:
                if st.button("主要商品"):
                    st.session_state.quick_select = ["RB", "AU", "CU", "M", "TA"]

        if selected_symbols:
            if st.button("🚀 开始下载", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                for i, symbol in enumerate(selected_symbols):
                    success, msg, count = download_symbol(symbol)
                    progress_bar.progress((i + 1) / len(selected_symbols))
                    if success:
                        st.write(f"✅ {msg} - {count}条")
                    else:
                        st.write(f"❌ {msg}")

    with tab2:
        st.subheader("下载分钟数据")
        st.info("💡 分钟数据来自新浪财经，约有最近1000根K线")

        categories = get_symbol_list_by_category()
        col1, col2 = st.columns(2)

        with col1:
            category_min = st.selectbox("选择类别 ", options=list(categories.keys()), key="min_cat")
            symbols_in_cat_min = categories[category_min]
            selected_symbols_min = st.multiselect(
                "选择品种 ",
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
            if st.button("🚀 开始下载分钟数据", type="primary"):
                for symbol in selected_symbols_min:
                    for period_name in selected_periods:
                        period = MINUTE_PERIODS[period_name]
                        success, msg, count = download_minute_symbol(symbol, period)
                        if success:
                            st.write(f"✅ {msg} - {count}条")
                        else:
                            st.write(f"❌ {msg}")

    with tab3:
        st.subheader("数据状态")
        if st.button("🔄 刷新"):
            st.cache_data.clear()

        df_status = get_data_status()
        df_with_data = df_status[df_status['record_count'] > 0].copy()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("已有数据品种", len(df_with_data))
        with col2:
            st.metric("无数据品种", len(df_status) - len(df_with_data))

        if len(df_with_data) > 0:
            df_display = df_with_data[['symbol', 'name', 'exchange', 'start_date', 'end_date', 'record_count']].copy()
            df_display.columns = ['代码', '名称', '交易所', '起始日期', '结束日期', '数据条数']
            st.dataframe(df_display, use_container_width=True, hide_index=True)


def main():
    render_header()

    # 侧边栏导航
    page = st.sidebar.radio(
        "🧭 导航",
        options=["📈 策略回测", "📥 数据管理"],
        index=0
    )

    if page == "📥 数据管理":
        render_data_management()
    else:
        # 策略回测页面
        col_left, col_right = st.columns([1, 2])

        with col_left:
            # 配置编辑器
            yaml_content = render_config_editor()

            # 标签页切换编辑模式
            edit_mode = st.radio(
                "编辑模式",
                options=["📝 YAML编辑", "🎛️ 可视化编辑"],
                horizontal=True,
                label_visibility="collapsed"
            )

            if edit_mode == "📝 YAML编辑":
                config = render_yaml_editor(yaml_content)
            else:
                try:
                    config = yaml_to_config(yaml_content)
                    config = render_visual_config(config)
                except:
                    config = None
                    st.error("配置解析失败")

            st.markdown("---")

            # 运行按钮
            st.markdown('<div class="run-btn">', unsafe_allow_html=True)
            run_btn = st.button("🚀 开始回测", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if run_btn:
                with st.spinner("正在回测..."):
                    result = run_backtest_from_config(config)
                    if result:
                        st.session_state['result'] = result[0]
                        st.session_state['df_data'] = result[1]
                        st.success(f"✅ 回测完成! 共 {len(result[0].trades)} 笔交易")

        with col_right:
            if 'result' in st.session_state:
                result = st.session_state['result']
                df_data = st.session_state.get('df_data')

                tabs = st.tabs(["📊 概览", "💹 资金曲线", "📋 交易记录", "📉 统计分析"])

                with tabs[0]:
                    render_metrics(result)

                with tabs[1]:
                    render_equity_chart(result)

                with tabs[2]:
                    render_trades_table(result)

                with tabs[3]:
                    render_statistics(result)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 100px 20px; color: #94a3b8;">
                    <h2>👈 配置策略后点击「开始回测」</h2>
                    <p>支持 YAML 配置文件，类似 banbot 风格</p>
                </div>
                """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
