# coding=utf-8
"""
期货策略回测系统
支持数据下载、多策略选择、时间周期选择和动态参数配置
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys

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

st.set_page_config(
    page_title="期货策略回测系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .win-trade { color: #00c853; font-weight: bold; }
    .loss-trade { color: #ff1744; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    .strategy-info {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .download-btn {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
    }
</style>
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


def render_strategy_params(strategy_class) -> dict:
    """动态渲染策略参数"""
    params = {}
    param_defs = strategy_class.get_params()

    # 按类型分组参数
    grouped_params = {
        '均线/周期参数': [],
        '风控参数': [],
        '仓位参数': [],
        '其他参数': []
    }

    for p in param_defs:
        if any(k in p.name for k in ['len', 'period', 'ma', 'ema', 'sma', 'fast', 'slow', 'bb', 'macd']):
            grouped_params['均线/周期参数'].append(p)
        elif any(k in p.name for k in ['stop', 'atr', 'risk', 'adx', 'drawdown', 'trigger', 'break']):
            grouped_params['风控参数'].append(p)
        elif any(k in p.name for k in ['capital', 'risk_rate', 'position', 'partial']):
            grouped_params['仓位参数'].append(p)
        else:
            grouped_params['其他参数'].append(p)

    # 渲染各组参数
    for group_name, group_params in grouped_params.items():
        if not group_params:
            continue

        with st.expander(group_name, expanded=True):
            for p in group_params:
                if p.param_type == 'int':
                    params[p.name] = st.slider(
                        p.label,
                        int(p.min_val) if p.min_val else 1,
                        int(p.max_val) if p.max_val else 100,
                        int(p.default),
                        int(p.step) if p.step else 1,
                        help=p.description
                    )
                elif p.param_type == 'float':
                    params[p.name] = st.slider(
                        p.label,
                        float(p.min_val) if p.min_val else 0.0,
                        float(p.max_val) if p.max_val else 1.0,
                        float(p.default),
                        float(p.step) if p.step else 0.01,
                        help=p.description
                    )
                elif p.param_type == 'bool':
                    params[p.name] = st.checkbox(
                        p.label,
                        value=bool(p.default),
                        help=p.description
                    )
                elif p.param_type == 'select' and p.options:
                    params[p.name] = st.selectbox(
                        p.label,
                        options=p.options,
                        index=p.options.index(p.default) if p.default in p.options else 0,
                        help=p.description
                    )

    return params


def render_data_management():
    """渲染数据管理页面"""
    st.header("📥 数据管理")

    tab1, tab2, tab3 = st.tabs(["日线数据", "分钟数据", "数据状态"])

    with tab1:
        st.subheader("下载期货数据")

        # 按类别选择品种
        categories = get_symbol_list_by_category()

        col1, col2 = st.columns(2)

        with col1:
            category = st.selectbox(
                "选择类别",
                options=list(categories.keys())
            )

            symbols_in_cat = categories[category]
            selected_symbols = st.multiselect(
                "选择品种",
                options=[s[0] for s in symbols_in_cat],
                format_func=lambda x: f"{x} - {FUTURES_SYMBOLS[x][0]}",
                default=[s[0] for s in symbols_in_cat[:2]] if symbols_in_cat else []
            )

        with col2:
            # 快捷选择
            st.write("**快捷选择:**")
            if st.button("全选当前类别"):
                selected_symbols = [s[0] for s in symbols_in_cat]

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("股指期货"):
                    selected_symbols = ["IF", "IH", "IC", "IM"]
            with col_b:
                if st.button("主要商品"):
                    selected_symbols = ["RB", "AU", "CU", "M", "TA"]

        st.markdown("---")

        # 下载按钮
        if selected_symbols:
            st.write(f"已选择 **{len(selected_symbols)}** 个品种: {', '.join(selected_symbols)}")

            if st.button("🚀 开始下载", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()

                results = {}
                for i, symbol in enumerate(selected_symbols):
                    status_text.text(f"正在下载 {symbol} ({i+1}/{len(selected_symbols)})...")
                    progress_bar.progress((i + 1) / len(selected_symbols))

                    success, msg, count = download_symbol(symbol)
                    results[symbol] = (success, msg, count)

                status_text.text("下载完成!")

                # 显示结果
                with results_container:
                    success_count = sum(1 for r in results.values() if r[0])
                    st.success(f"成功下载 {success_count}/{len(results)} 个品种")

                    for symbol, (success, msg, count) in results.items():
                        if success:
                            st.write(f"✅ {msg} - {count}条数据")
                        else:
                            st.write(f"❌ {msg}")
        else:
            st.info("请选择要下载的品种")

    with tab2:
        st.subheader("下载分钟数据")
        st.info("💡 分钟数据来自新浪财经，约有最近1000根K线")

        # 按类别选择品种
        categories = get_symbol_list_by_category()

        col1, col2 = st.columns(2)

        with col1:
            category_min = st.selectbox(
                "选择类别 ",
                options=list(categories.keys()),
                key="minute_category"
            )

            symbols_in_cat_min = categories[category_min]
            selected_symbols_min = st.multiselect(
                "选择品种 ",
                options=[s[0] for s in symbols_in_cat_min],
                format_func=lambda x: f"{x} - {FUTURES_SYMBOLS[x][0]}",
                default=[],
                key="minute_symbols"
            )

        with col2:
            # 选择周期
            st.write("**选择周期:**")
            selected_periods = st.multiselect(
                "K线周期",
                options=list(MINUTE_PERIODS.keys()),
                default=["60分钟"],
                key="minute_periods"
            )

        st.markdown("---")

        # 下载按钮
        if selected_symbols_min and selected_periods:
            total_tasks = len(selected_symbols_min) * len(selected_periods)
            st.write(f"已选择 **{len(selected_symbols_min)}** 个品种, **{len(selected_periods)}** 个周期, 共 **{total_tasks}** 个任务")

            if st.button("🚀 开始下载分钟数据", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()

                results = {}
                task_idx = 0
                for symbol in selected_symbols_min:
                    for period_name in selected_periods:
                        period = MINUTE_PERIODS[period_name]
                        status_text.text(f"正在下载 {symbol} {period_name} ({task_idx+1}/{total_tasks})...")
                        progress_bar.progress((task_idx + 1) / total_tasks)

                        success, msg, count = download_minute_symbol(symbol, period)
                        results[f"{symbol}_{period}"] = (success, msg, count)
                        task_idx += 1

                status_text.text("下载完成!")

                # 显示结果
                with results_container:
                    success_count = sum(1 for r in results.values() if r[0])
                    st.success(f"成功下载 {success_count}/{len(results)} 个任务")

                    for key, (success, msg, count) in results.items():
                        if success:
                            st.write(f"✅ {msg} - {count}条数据")
                        else:
                            st.write(f"❌ {msg}")
        else:
            st.info("请选择要下载的品种和周期")

    with tab3:
        st.subheader("数据状态")

        if st.button("🔄 刷新数据状态"):
            st.cache_data.clear()

        # 日线数据状态
        st.write("### 📊 日线数据")
        df_status = get_data_status()

        # 筛选有数据的品种
        df_with_data = df_status[df_status['record_count'] > 0].copy()
        df_no_data = df_status[df_status['record_count'] == 0].copy()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("已有数据品种", len(df_with_data))
        with col2:
            st.metric("无数据品种", len(df_no_data))

        if len(df_with_data) > 0:
            df_display = df_with_data[['symbol', 'name', 'exchange', 'start_date', 'end_date', 'record_count']].copy()
            df_display.columns = ['代码', '名称', '交易所', '起始日期', '结束日期', '数据条数']
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        # 分钟数据状态
        st.write("### ⏱️ 分钟数据")
        df_minute_status = get_minute_data_status()

        if len(df_minute_status) > 0:
            # 添加品种名称
            df_minute_status['name'] = df_minute_status['symbol'].apply(
                lambda x: FUTURES_SYMBOLS.get(x, ('未知',))[0]
            )
            df_minute_display = df_minute_status[['symbol', 'name', 'period', 'start_time', 'end_time', 'record_count']].copy()
            df_minute_display.columns = ['代码', '名称', '周期(分钟)', '起始时间', '结束时间', '数据条数']
            st.dataframe(df_minute_display, use_container_width=True, hide_index=True)
        else:
            st.info("暂无分钟数据，请先下载")


def render_backtest_page():
    """渲染回测页面"""
    st.header("📊 策略回测")

    # 左右布局
    col_config, col_result = st.columns([1, 2])

    with col_config:
        st.subheader("⚙️ 回测配置")

        # ========== 策略选择 ==========
        strategies = get_all_strategies()
        strategy_names = list(strategies.keys())
        strategy_display = {k: v.display_name for k, v in strategies.items()}

        selected_strategy_name = st.selectbox(
            "🎯 选择策略",
            options=strategy_names,
            format_func=lambda x: f"{strategy_display[x]} ({x})"
        )

        strategy_class = strategies[selected_strategy_name]

        # 显示策略信息
        with st.expander("📖 策略说明", expanded=False):
            st.markdown(f"**{strategy_class.display_name}**")
            st.markdown(f"*版本: {strategy_class.version}*")
            st.markdown(strategy_class.description)

        st.markdown("---")

        # ========== 品种选择 ==========
        st.write("**📌 品种选择**")

        # 从数据库获取有数据的品种
        df_status = get_data_status()
        symbols_with_data = df_status[df_status['record_count'] > 0]['symbol'].tolist()

        if not symbols_with_data:
            st.warning("没有数据，请先在「数据管理」页面下载数据")
            return None

        symbol = st.selectbox(
            "选择品种",
            options=symbols_with_data,
            format_func=lambda x: f"{x} - {FUTURES_SYMBOLS.get(x, ('未知',))[0]}"
        )

        # 获取该品种的数据范围
        symbol_info = df_status[df_status['symbol'] == symbol].iloc[0]
        data_start = symbol_info['start_date']
        data_end = symbol_info['end_date']

        st.caption(f"数据范围: {data_start} ~ {data_end}")

        st.markdown("---")

        # ========== 时间周期 ==========
        st.write("**⏱️ 时间周期**")
        time_period = st.selectbox(
            "K线周期",
            options=["日线", "周线", "月线", "60分钟", "30分钟", "15分钟", "5分钟"],
            index=0
        )

        st.markdown("---")

        # ========== 回测时间范围 ==========
        st.write("**📅 回测时间范围**")

        col_start, col_end = st.columns(2)

        # 解析数据范围日期
        try:
            min_date = datetime.strptime(data_start, '%Y-%m-%d').date()
            max_date = datetime.strptime(data_end, '%Y-%m-%d').date()
        except:
            min_date = datetime(2010, 1, 1).date()
            max_date = datetime.now().date()

        with col_start:
            start_date = st.date_input(
                "起始日期",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )

        with col_end:
            end_date = st.date_input(
                "结束日期",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )

        st.markdown("---")

        # ========== 资金设置 ==========
        st.write("**💰 资金设置**")
        initial_capital = st.number_input(
            "初始资金 (元)",
            min_value=100000,
            max_value=100000000,
            value=1000000,
            step=100000
        )

        st.markdown("---")

        # ========== 策略参数 ==========
        st.write("**🔧 策略参数**")
        params = render_strategy_params(strategy_class)

        st.markdown("---")

        # ========== 开始回测按钮 ==========
        run_backtest_btn = st.button(
            "🚀 开始回测",
            type="primary",
            use_container_width=True
        )

        return {
            'symbol': symbol,
            'strategy_class': strategy_class,
            'params': params,
            'initial_capital': initial_capital,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'time_period': time_period,
            'run_backtest': run_backtest_btn
        }

    return None


def run_backtest_and_display(config, result_container):
    """运行回测并显示结果"""
    with result_container:
        with st.spinner(f"正在使用 {config['strategy_class'].display_name} 策略回测..."):
            try:
                time_period = config['time_period']

                # 根据周期类型加载数据
                if time_period in ["5分钟", "15分钟", "30分钟", "60分钟"]:
                    # 加载分钟数据
                    period_map = {"5分钟": "5", "15分钟": "15", "30分钟": "30", "60分钟": "60"}
                    period = period_map[time_period]
                    df_data = load_minute_from_database(
                        config['symbol'],
                        period,
                        config['start_date'],
                        config['end_date']
                    )
                    if len(df_data) == 0:
                        st.error(f"没有 {time_period} 数据，请先在「数据管理」页面下载分钟数据")
                        return
                else:
                    # 加载日线数据
                    df_data = load_from_database(
                        config['symbol'],
                        config['start_date'],
                        config['end_date']
                    )

                    if len(df_data) == 0:
                        st.error("没有数据，请先下载数据")
                        return

                    # 重采样到指定周期 (周线/月线)
                    df_data = resample_data(df_data, time_period)

                st.info(f"数据: {len(df_data)} 条 ({config['start_date']} ~ {config['end_date']}) - {config['time_period']}")

                # 创建策略实例
                strategy_instance = config['strategy_class'](config['params'])

                # 运行回测
                result = run_backtest_with_strategy(
                    df_data,
                    config['symbol'],
                    strategy_instance,
                    config['initial_capital']
                )

                # 保存结果到session
                st.session_state['result'] = result
                st.session_state['df_data'] = df_data
                st.session_state['params'] = config['params']
                st.session_state['strategy_class'] = config['strategy_class']

                st.success(f"✅ 回测完成! 共 {len(result.trades)} 笔交易")

            except Exception as e:
                st.error(f"回测失败: {e}")
                import traceback
                st.code(traceback.format_exc())


def render_overview(result):
    """渲染概览页"""
    st.subheader("📊 回测概览")

    # 顶部指标卡片
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            "总收益",
            f"¥{result.total_pnl:,.0f}",
            f"{result.total_return_pct:+.2f}%"
        )
    with col2:
        st.metric(
            "年化收益",
            f"{result.annual_return_pct:.2f}%"
        )
    with col3:
        st.metric(
            "最大回撤",
            f"{result.max_drawdown_pct:.2f}%",
            f"¥{result.max_drawdown_val:,.0f}"
        )
    with col4:
        st.metric(
            "夏普比率",
            f"{result.sharpe_ratio:.2f}"
        )
    with col5:
        st.metric(
            "胜率",
            f"{result.win_rate:.1f}%",
            f"{len([t for t in result.trades if t.pnl > 0])}/{len(result.trades)}"
        )
    with col6:
        st.metric(
            "盈亏比",
            f"{result.profit_factor:.2f}"
        )

    st.markdown("---")

    # 详细指标
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**💰 收益指标**")
        st.write(f"初始资金: ¥{result.initial_capital:,.0f}")
        st.write(f"期末资金: ¥{result.final_capital:,.0f}")
        st.write(f"总盈亏: ¥{result.total_pnl:,.0f}")
        st.write(f"总收益率: {result.total_return_pct:.2f}%")
        st.write(f"年化收益: {result.annual_return_pct:.2f}%")
        st.write(f"总手续费: ¥{result.total_commission:,.0f}")

    with col2:
        st.write("**📉 风险指标**")
        st.write(f"最大回撤: {result.max_drawdown_pct:.2f}%")
        st.write(f"回撤金额: ¥{result.max_drawdown_val:,.0f}")
        st.write(f"夏普比率: {result.sharpe_ratio:.2f}")
        st.write(f"索提诺比率: {result.sortino_ratio:.2f}")
        st.write(f"卡尔玛比率: {result.calmar_ratio:.2f}")

    with col3:
        st.write("**📈 交易指标**")
        st.write(f"总交易数: {len(result.trades)}")
        st.write(f"胜率: {result.win_rate:.1f}%")
        st.write(f"盈亏比: {result.profit_factor:.2f}")
        st.write(f"平均盈利: ¥{result.avg_win:,.0f}")
        st.write(f"平均亏损: ¥{result.avg_loss:,.0f}")
        st.write(f"平均持仓: {result.avg_holding_days:.1f}天")


def render_equity_chart(result):
    """渲染资金曲线"""
    st.subheader("💹 资金曲线")

    df = result.equity_curve

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('账户净值', '回撤')
    )

    # 资金曲线
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['equity'],
            name='账户净值',
            line=dict(color='#2196F3', width=2),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.1)'
        ),
        row=1, col=1
    )

    # 标记交易点
    for trade in result.trades:
        color = '#4CAF50' if trade.pnl > 0 else '#F44336'
        # 入场
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_time],
                y=[trade.capital_before if trade.capital_before > 0 else result.initial_capital],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='#2196F3'),
                name='入场',
                showlegend=False,
                hovertemplate=f"入场: {trade.entry_price:.1f}<br>手数: {trade.volume}"
            ),
            row=1, col=1
        )
        # 出场
        fig.add_trace(
            go.Scatter(
                x=[trade.exit_time],
                y=[trade.capital_after],
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color=color),
                name='出场',
                showlegend=False,
                hovertemplate=f"出场: {trade.exit_price:.1f}<br>盈亏: ¥{trade.pnl:,.0f}"
            ),
            row=1, col=1
        )

    # 回撤曲线
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=-df['drawdown_pct'],
            name='回撤',
            line=dict(color='#F44336', width=1),
            fill='tozeroy',
            fillcolor='rgba(244, 67, 54, 0.3)'
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        hovermode='x unified',
        showlegend=True
    )

    fig.update_yaxes(title_text="净值 (元)", row=1, col=1)
    fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_trades_table(result):
    """渲染交易列表"""
    st.subheader("📋 交易记录")

    if not result.trades:
        st.warning("没有交易记录")
        return

    # 转换为DataFrame
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

    # 筛选器
    col1, col2 = st.columns(2)
    with col1:
        result_filter = st.multiselect(
            "筛选结果",
            options=['盈', '亏'],
            default=['盈', '亏']
        )
    with col2:
        exit_tags = df_trades['出场原因'].unique().tolist()
        tag_filter = st.multiselect(
            "筛选出场原因",
            options=exit_tags,
            default=exit_tags
        )

    # 应用筛选
    df_filtered = df_trades[
        (df_trades['结果'].isin(result_filter)) &
        (df_trades['出场原因'].isin(tag_filter))
    ]

    st.dataframe(df_filtered, use_container_width=True, hide_index=True)

    # 下载按钮
    csv = df_filtered.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        "📥 下载交易记录",
        csv,
        "trades.csv",
        "text/csv"
    )


def render_kline_with_trades(result, df_data):
    """渲染K线图并标记交易"""
    st.subheader("📈 K线交易图")

    if not result.trades:
        st.warning("没有交易记录")
        return

    if df_data is None or len(df_data) == 0:
        st.warning("没有K线数据")
        return

    # 筛选器
    col1, col2, col3 = st.columns(3)

    with col1:
        result_filter = st.multiselect(
            "筛选结果",
            options=['盈利', '亏损'],
            default=['盈利', '亏损'],
            key="kline_result_filter"
        )

    with col2:
        exit_tags = list(set([t.exit_tag for t in result.trades]))
        tag_filter = st.multiselect(
            "筛选出场原因",
            options=exit_tags,
            default=exit_tags,
            key="kline_tag_filter"
        )

    with col3:
        # 选择要查看的交易
        trade_options = [f"#{t.trade_id+1} {t.entry_time.strftime('%m-%d')}→{t.exit_time.strftime('%m-%d') if t.exit_time else ''} {'盈' if t.pnl > 0 else '亏'}{abs(t.pnl_pct):.1f}%"
                        for t in result.trades]
        selected_trade_idx = st.selectbox(
            "跳转到交易",
            options=range(len(trade_options)),
            format_func=lambda x: trade_options[x],
            key="kline_trade_select"
        )

    # 筛选交易
    filtered_trades = [t for t in result.trades
                      if (('盈利' in result_filter and t.pnl > 0) or ('亏损' in result_filter and t.pnl <= 0))
                      and t.exit_tag in tag_filter]

    st.write(f"显示 **{len(filtered_trades)}** / {len(result.trades)} 笔交易")

    # 计算价格范围，用于固定Y轴
    price_min = df_data['low'].min()
    price_max = df_data['high'].max()
    price_range = price_max - price_min
    y_min = price_min - price_range * 0.05  # 下方留5%空间
    y_max = price_max + price_range * 0.08  # 上方留8%空间（给标记文字）

    # 创建K线图
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.15, 0.15],
        subplot_titles=('', '', '')  # 移除子图标题，节省空间
    )

    # K线图
    fig.add_trace(
        go.Candlestick(
            x=df_data['time'],
            open=df_data['open'],
            high=df_data['high'],
            low=df_data['low'],
            close=df_data['close'],
            name='K线',
            increasing_line_color='#EF5350',  # 红涨
            decreasing_line_color='#26A69A',  # 绿跌
            increasing_fillcolor='#EF5350',
            decreasing_fillcolor='#26A69A'
        ),
        row=1, col=1
    )

    # 成交量
    colors = ['#EF5350' if close >= open else '#26A69A'
              for close, open in zip(df_data['close'], df_data['open'])]
    fig.add_trace(
        go.Bar(
            x=df_data['time'],
            y=df_data['volume'],
            name='成交量',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )

    # 持仓盈亏曲线（每笔交易期间）
    holding_pnl = []
    holding_time = []
    for t in filtered_trades:
        # 获取持仓期间的数据
        mask = (df_data['time'] >= t.entry_time) & (df_data['time'] <= t.exit_time)
        trade_data = df_data[mask]
        for _, row in trade_data.iterrows():
            pnl_pct = (row['close'] - t.entry_price) / t.entry_price * 100
            holding_pnl.append(pnl_pct)
            holding_time.append(row['time'])

    if holding_pnl:
        fig.add_trace(
            go.Scatter(
                x=holding_time,
                y=holding_pnl,
                mode='lines',
                name='持仓盈亏%',
                line=dict(color='#FF9800', width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 152, 0, 0.2)'
            ),
            row=3, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    # 标记交易入场和出场
    for t in filtered_trades:
        is_win = t.pnl > 0

        # 入场标记 - 在K线下方
        entry_low = df_data[df_data['time'] == t.entry_time]['low'].values
        entry_y = entry_low[0] * 0.995 if len(entry_low) > 0 else t.entry_price

        fig.add_trace(
            go.Scatter(
                x=[t.entry_time],
                y=[entry_y],
                mode='markers+text',
                marker=dict(
                    symbol='triangle-up',
                    size=16,
                    color='#2196F3',
                    line=dict(color='white', width=1)
                ),
                text=[f'买{t.volume}手'],
                textposition='bottom center',
                textfont=dict(size=10, color='#2196F3'),
                name=f'入场#{t.trade_id+1}',
                showlegend=False,
                hovertemplate=f"<b>入场 #{t.trade_id+1}</b><br>" +
                             f"时间: {t.entry_time.strftime('%Y-%m-%d')}<br>" +
                             f"价格: {t.entry_price:.2f}<br>" +
                             f"手数: {t.volume}<extra></extra>"
            ),
            row=1, col=1
        )

        # 出场标记 - 在K线上方
        if t.exit_time:
            exit_high = df_data[df_data['time'] == t.exit_time]['high'].values
            exit_y = exit_high[0] * 1.005 if len(exit_high) > 0 else t.exit_price

            exit_color = '#4CAF50' if is_win else '#F44336'

            fig.add_trace(
                go.Scatter(
                    x=[t.exit_time],
                    y=[exit_y],
                    mode='markers+text',
                    marker=dict(
                        symbol='triangle-down',
                        size=16,
                        color=exit_color,
                        line=dict(color='white', width=1)
                    ),
                    text=[f'{t.pnl_pct:+.1f}%'],
                    textposition='top center',
                    textfont=dict(size=10, color=exit_color, weight='bold'),
                    name=f'出场#{t.trade_id+1}',
                    showlegend=False,
                    hovertemplate=f"<b>出场 #{t.trade_id+1}</b><br>" +
                                 f"时间: {t.exit_time.strftime('%Y-%m-%d')}<br>" +
                                 f"价格: {t.exit_price:.2f}<br>" +
                                 f"盈亏: ¥{t.pnl:+,.0f} ({t.pnl_pct:+.2f}%)<br>" +
                                 f"原因: {t.exit_tag}<br>" +
                                 f"持仓: {t.holding_days}天<extra></extra>"
                ),
                row=1, col=1
            )

            # 连接入场和出场的线
            fig.add_trace(
                go.Scatter(
                    x=[t.entry_time, t.exit_time],
                    y=[t.entry_price, t.exit_price],
                    mode='lines',
                    line=dict(
                        color=exit_color,
                        width=2,
                        dash='dot'
                    ),
                    opacity=0.6,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

            # 持仓区间背景色 - 使用 shape 并限制在价格范围内
            fig.add_shape(
                type="rect",
                x0=t.entry_time, x1=t.exit_time,
                y0=y_min, y1=y_max,
                fillcolor='rgba(76, 175, 80, 0.15)' if is_win else 'rgba(244, 67, 54, 0.15)',
                layer='below',
                line_width=0,
                row=1, col=1
            )

    # 如果选择了特定交易，聚焦到该交易
    if selected_trade_idx is not None and selected_trade_idx < len(result.trades):
        selected_trade = result.trades[selected_trade_idx]
        # 计算显示范围（交易前后各30根K线）
        trade_start = selected_trade.entry_time
        trade_end = selected_trade.exit_time if selected_trade.exit_time else trade_start

        # 找到对应的索引
        try:
            start_idx = df_data[df_data['time'] <= trade_start].index[-1] - 30
            end_idx = df_data[df_data['time'] >= trade_end].index[0] + 30
            start_idx = max(0, start_idx)
            end_idx = min(len(df_data) - 1, end_idx)

            x_start = df_data.iloc[start_idx]['time']
            x_end = df_data.iloc[end_idx]['time']

            # 计算这段时间的价格范围
            visible_data = df_data.iloc[start_idx:end_idx+1]
            vis_min = visible_data['low'].min()
            vis_max = visible_data['high'].max()
            vis_range = vis_max - vis_min
            y_min = vis_min - vis_range * 0.05
            y_max = vis_max + vis_range * 0.10

            fig.update_xaxes(range=[x_start, x_end])
            fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
        except:
            pass

    # 更新布局
    fig.update_layout(
        height=700,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=30, b=30)
    )

    # 固定K线图Y轴范围，防止被挤压
    fig.update_yaxes(title_text="价格", row=1, col=1, range=[y_min, y_max], fixedrange=False)
    fig.update_yaxes(title_text="量", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # 显示选中交易的详情
    if selected_trade_idx is not None and selected_trade_idx < len(result.trades):
        t = result.trades[selected_trade_idx]
        st.markdown("---")
        st.write(f"### 交易 #{t.trade_id+1} 详情")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("入场时间", t.entry_time.strftime('%Y-%m-%d'))
            st.metric("入场价格", f"{t.entry_price:.2f}")
        with col2:
            st.metric("出场时间", t.exit_time.strftime('%Y-%m-%d') if t.exit_time else '-')
            st.metric("出场价格", f"{t.exit_price:.2f}" if t.exit_price else '-')
        with col3:
            st.metric("持仓天数", f"{t.holding_days}天")
            st.metric("交易手数", f"{t.volume}手")
        with col4:
            pnl_color = "normal" if t.pnl > 0 else "inverse"
            st.metric("盈亏金额", f"¥{t.pnl:+,.0f}", delta=f"{t.pnl_pct:+.2f}%")
            st.metric("出场原因", t.exit_tag)


def render_statistics(result):
    """渲染统计分析"""
    st.subheader("📊 统计分析")

    if not result.trades:
        st.warning("没有交易记录")
        return

    col1, col2 = st.columns(2)

    with col1:
        # 出场原因统计
        if result.exit_tag_stats is not None:
            st.write("**出场原因统计**")
            df_exit = result.exit_tag_stats.reset_index()
            df_exit.columns = ['出场原因', '次数', '总盈亏', '平均盈亏', '平均收益%']
            st.dataframe(df_exit, use_container_width=True, hide_index=True)

    with col2:
        # 收益分布
        st.write("**收益分布**")
        pnl_list = [t.pnl for t in result.trades]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=pnl_list,
            nbinsx=20,
            marker_color='#2196F3'
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(
            height=300,
            xaxis_title='盈亏金额 (元)',
            yaxis_title='次数',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # 盈亏对比
    st.markdown("---")
    wins = [t.pnl for t in result.trades if t.pnl > 0]
    losses = [t.pnl for t in result.trades if t.pnl <= 0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("盈利交易", f"{len(wins)}笔")
    with col2:
        st.metric("盈利总额", f"¥{sum(wins):,.0f}" if wins else "¥0")
    with col3:
        st.metric("亏损交易", f"{len(losses)}笔")
    with col4:
        st.metric("亏损总额", f"¥{sum(losses):,.0f}" if losses else "¥0")


def main():
    st.title("📊 期货策略回测系统")

    # 主导航
    page = st.sidebar.radio(
        "导航",
        options=["📈 策略回测", "📥 数据管理"],
        index=0
    )

    if page == "📥 数据管理":
        render_data_management()

    else:  # 策略回测
        config = render_backtest_page()

        # 结果显示区域
        if config is None:
            return

        result_container = st.container()

        # 运行回测
        if config['run_backtest']:
            run_backtest_and_display(config, result_container)

        # 显示已有结果
        if 'result' in st.session_state:
            result = st.session_state['result']
            df_data = st.session_state.get('df_data', None)

            with result_container:
                # 标签页
                tabs = st.tabs(["📊 概览", "📈 K线交易图", "💹 资金曲线", "📋 交易记录", "📉 统计分析"])

                with tabs[0]:
                    render_overview(result)

                with tabs[1]:
                    render_kline_with_trades(result, df_data)

                with tabs[2]:
                    render_equity_chart(result)

                with tabs[3]:
                    render_trades_table(result)

                with tabs[4]:
                    render_statistics(result)


if __name__ == '__main__':
    main()
