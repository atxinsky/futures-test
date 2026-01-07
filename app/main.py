# coding=utf-8
"""
专业交易系统 Web界面
主入口文件 - 集成完整回测功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入回测相关模块
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

# 实盘交易模块
try:
    from app.live_trading import render_live_trading_page
    HAS_LIVE_TRADING = True
except ImportError:
    HAS_LIVE_TRADING = False

# 模拟交易模块
try:
    from app.sim_trading import render_sim_trading_page
    HAS_SIM_TRADING = True
except ImportError:
    HAS_SIM_TRADING = False

import json

# TqSdk配置文件路径
TQ_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tq_config.json")


def load_tq_config_for_settings() -> dict:
    """加载TqSdk配置（用于系统设置）"""
    if os.path.exists(TQ_CONFIG_FILE):
        try:
            with open(TQ_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {
        'tq_user': '',
        'tq_password': '',
        'sim_mode': True,
        'broker_id': '',
        'td_account': '',
        'td_password': '',
        'default_symbols': ['RB', 'AU', 'IF'],
        'initial_capital': 100000,
        'risk_config': {
            'max_position_per_symbol': 10,
            'max_daily_loss': 0.05,
            'max_drawdown': 0.15
        }
    }


def save_tq_config_for_settings(config: dict):
    """保存TqSdk配置（用于系统设置）"""
    with open(TQ_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def test_tq_connection_settings(tq_user: str, tq_password: str):
    """测试天勤连接"""
    if not tq_user or not tq_password:
        st.error("请输入天勤账号和密码")
        return

    try:
        from tqsdk import TqApi, TqAuth

        with st.spinner("正在连接天勤..."):
            auth = TqAuth(tq_user, tq_password)
            api = TqApi(auth=auth)

            # 获取行情测试
            quote = api.get_quote("SHFE.rb2505")
            api.wait_update()

            api.close()

        st.success(f"连接成功! 测试行情: RB2505 最新价 {quote.last_price}")

    except ImportError:
        st.error("TqSdk未安装，请执行: pip install tqsdk")
    except Exception as e:
        st.error(f"连接失败: {e}")


# ============ 回测辅助函数 ============

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


def render_strategy_params(strategy_class, loaded_params=None, config_key="") -> dict:
    """动态渲染策略参数"""
    params = {}
    param_defs = strategy_class.get_params()
    loaded_params = loaded_params or {}

    # 按类型分组参数
    grouped_params = {
        '均线/周期参数': [],
        '风控参数': [],
        '仓位参数': [],
        '其他参数': []
    }

    for p in param_defs:
        if any(k in p.name for k in ['len', 'period', 'ma', 'ema', 'sma', 'fast', 'slow', 'bb', 'macd', 'chop', 'vol']):
            grouped_params['均线/周期参数'].append(p)
        elif any(k in p.name for k in ['stop', 'atr', 'risk', 'adx', 'drawdown', 'trigger', 'break', 'partial', 'full']):
            grouped_params['风控参数'].append(p)
        elif any(k in p.name for k in ['capital', 'risk_rate', 'position']):
            grouped_params['仓位参数'].append(p)
        else:
            grouped_params['其他参数'].append(p)

    # 渲染各组参数
    for group_name, group_params in grouped_params.items():
        if not group_params:
            continue

        with st.expander(group_name, expanded=True):
            cols = st.columns(3)
            for i, p in enumerate(group_params):
                with cols[i % 3]:
                    if p.param_type == 'int':
                        default_val = int(loaded_params.get(p.name, p.default))
                        params[p.name] = st.number_input(
                            p.label,
                            min_value=int(p.min_val) if p.min_val else 1,
                            max_value=int(p.max_val) if p.max_val else 100,
                            value=default_val,
                            step=int(p.step) if p.step else 1,
                            help=p.description,
                            key=f"param_{p.name}_{config_key}"
                        )
                    elif p.param_type == 'float':
                        default_val = float(loaded_params.get(p.name, p.default))
                        params[p.name] = st.number_input(
                            p.label,
                            min_value=float(p.min_val) if p.min_val else 0.0,
                            max_value=float(p.max_val) if p.max_val else 100.0,
                            value=default_val,
                            step=float(p.step) if p.step else 0.01,
                            format="%.2f",
                            help=p.description,
                            key=f"param_{p.name}_{config_key}"
                        )
                    elif p.param_type == 'bool':
                        default_val = bool(loaded_params.get(p.name, p.default))
                        params[p.name] = st.checkbox(
                            p.label,
                            value=default_val,
                            help=p.description,
                            key=f"param_{p.name}_{config_key}"
                        )

    return params


def render_backtest_config():
    """渲染回测配置页面 - 左右并排布局"""

    # ========== 加载配置和策略 ==========
    configs = list_configs()
    config_options = ["手动配置"] + configs

    strategies = get_all_strategies()
    strategy_names = list(strategies.keys())
    strategy_display = {k: v.display_name for k, v in strategies.items()}
    default_idx = strategy_names.index('brother2v6') if 'brother2v6' in strategy_names else 0

    df_status = get_data_status()
    symbols_with_data = df_status[df_status['record_count'] > 0]['symbol'].tolist()

    if not symbols_with_data:
        st.warning("没有数据，请先在「数据管理」页面下载数据")
        return None

    # ========== 三列布局：基础设置 | 策略参数 | 合约信息 ==========
    col_settings, col_params, col_info = st.columns([1, 1.5, 0.8])

    # ========== 左列：基础设置 ==========
    with col_settings:
        st.subheader("基础设置")

        # 配置文件选择
        c1, c2 = st.columns([4, 1])
        with c1:
            selected_config = st.selectbox("配置文件", options=config_options, key="config_select")

        # 检测配置是否变化
        if 'last_config' not in st.session_state:
            st.session_state.last_config = selected_config
        if st.session_state.last_config != selected_config:
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith('param_')]
            for k in keys_to_delete:
                del st.session_state[k]
            st.session_state.last_config = selected_config
            st.rerun()

        with c2:
            st.write("")
            if selected_config != "手动配置":
                if st.button("删除", key="del_config"):
                    delete_config(selected_config)
                    st.rerun()

        # 加载配置文件内容
        loaded_params = {}
        loaded_strategy = None
        loaded_symbol = None
        loaded_timeframe = None
        loaded_capital = 1000000

        if selected_config != "手动配置":
            config = load_config(selected_config)
            loaded_params = config.get('run_policy', {}).get('params', {})
            loaded_strategy = config.get('run_policy', {}).get('name', None)
            loaded_timeframe = config.get('run_policy', {}).get('timeframes', None)
            loaded_capital = config.get('initial_capital', 1000000)
            pairs = config.get('pairs', [])
            if pairs:
                loaded_symbol = pairs[0]

        # 策略选择
        strategy_idx = default_idx
        if loaded_strategy and loaded_strategy in strategy_names:
            strategy_idx = strategy_names.index(loaded_strategy)

        selected_strategy_name = st.selectbox(
            "选择策略",
            options=strategy_names,
            index=strategy_idx,
            format_func=lambda x: f"{strategy_display[x]}"
        )
        strategy_class = strategies[selected_strategy_name]

        # 品种选择
        symbol_idx = 0
        if loaded_symbol and loaded_symbol in symbols_with_data:
            symbol_idx = symbols_with_data.index(loaded_symbol)

        symbol = st.selectbox(
            "选择品种",
            options=symbols_with_data,
            index=symbol_idx,
            format_func=lambda x: f"{x} - {FUTURES_SYMBOLS.get(x, ('未知',))[0]}"
        )

        # 时间周期
        timeframe_options = ["日线", "周线", "月线", "60分钟", "30分钟", "15分钟", "5分钟"]
        timeframe_idx = 0
        if loaded_timeframe and loaded_timeframe in timeframe_options:
            timeframe_idx = timeframe_options.index(loaded_timeframe)

        time_period = st.selectbox(
            "K线周期",
            options=timeframe_options,
            index=timeframe_idx
        )

        # 回测时间
        symbol_info = df_status[df_status['symbol'] == symbol].iloc[0]
        data_start = symbol_info['start_date']
        data_end = symbol_info['end_date']

        try:
            min_date = datetime.strptime(data_start, '%Y-%m-%d').date()
            max_date = datetime.strptime(data_end, '%Y-%m-%d').date()
        except:
            min_date = datetime(2010, 1, 1).date()
            max_date = datetime.now().date()

        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("起始日期", value=min_date, min_value=min_date, max_value=max_date)
        with c2:
            end_date = st.date_input("结束日期", value=max_date, min_value=min_date, max_value=max_date)

        # 资金设置
        initial_capital = st.number_input(
            "初始资金",
            min_value=100000,
            max_value=100000000,
            value=int(loaded_capital),
            step=100000
        )

        # 开始回测按钮
        st.markdown("---")
        run_backtest_btn = st.button("开始回测", type="primary", use_container_width=True)

        # 保存配置
        with st.expander("保存配置"):
            save_name = st.text_input("名称", value=f"{selected_strategy_name}_{symbol}")
            if st.button("保存"):
                cfg = {
                    'name': save_name,
                    'initial_capital': initial_capital,
                    'time_start': start_date.strftime('%Y%m%d'),
                    'time_end': end_date.strftime('%Y%m%d'),
                    'run_policy': {'name': selected_strategy_name, 'timeframes': time_period, 'params': params},
                    'pairs': [symbol]
                }
                save_config(f"{save_name}.yml", cfg)
                st.success(f"已保存!")

    # ========== 中列：策略参数 ==========
    with col_params:
        st.subheader(f"{strategy_class.display_name} 参数")
        params = render_strategy_params(strategy_class, loaded_params, selected_config)

    # ========== 右列：合约信息 ==========
    with col_info:
        st.subheader("合约规格")

        inst = get_instrument(symbol)
        if inst:
            st.metric("品种", f"{inst['name']}")
            st.metric("合约乘数", f"{inst['multiplier']}")
            st.metric("最小变动", f"{inst['price_tick']}")
            st.metric("保证金率", f"{inst['margin_rate']*100:.0f}%")
            if inst['commission_fixed'] > 0:
                st.metric("手续费", f"{inst['commission_fixed']}元/手")
            else:
                st.metric("手续费率", f"{inst['commission_rate']*10000:.2f}%%")
            st.metric("交易所", inst['exchange'])

            st.markdown("---")
            st.caption(f"1手价值 ≈ 价格×{inst['multiplier']}")

        # 数据信息
        st.markdown("---")
        st.write("**数据范围**")
        st.caption(f"{data_start} ~ {data_end}")
        st.caption(f"共 {symbol_info['record_count']:,} 条")

    st.markdown("---")

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


def run_backtest_and_display(config, result_container):
    """运行回测并显示结果"""
    with result_container:
        with st.spinner(f"正在使用 {config['strategy_class'].display_name} 策略回测..."):
            try:
                time_period = config['time_period']

                if time_period in ["5分钟", "15分钟", "30分钟", "60分钟"]:
                    period_map = {"5分钟": "5", "15分钟": "15", "30分钟": "30", "60分钟": "60"}
                    period = period_map[time_period]
                    df_data = load_minute_from_database(
                        config['symbol'], period, config['start_date'], config['end_date']
                    )
                    if len(df_data) == 0:
                        st.error(f"没有 {time_period} 数据，请先下载分钟数据")
                        return
                else:
                    df_data = load_from_database(config['symbol'], config['start_date'], config['end_date'])
                    if len(df_data) == 0:
                        st.error("没有数据，请先下载数据")
                        return
                    df_data = resample_data(df_data, time_period)

                st.info(f"数据: {len(df_data)} 条 ({config['start_date']} ~ {config['end_date']}) - {config['time_period']}")

                strategy_instance = config['strategy_class'](config['params'])
                result = run_backtest_with_strategy(df_data, config['symbol'], strategy_instance, config['initial_capital'])

                st.session_state['backtest_result'] = result
                st.session_state['backtest_df_data'] = df_data
                st.session_state['backtest_params'] = config['params']
                st.session_state['backtest_strategy_class'] = config['strategy_class']

                st.success(f"回测完成! 共 {len(result.trades)} 笔交易")

            except Exception as e:
                st.error(f"回测失败: {e}")
                import traceback
                st.code(traceback.format_exc())


def render_backtest_overview(result):
    """渲染回测概览页"""
    st.subheader("回测概览")

    # 显示合约规格信息
    inst = get_instrument(result.symbol)
    if inst:
        with st.expander(f"{result.symbol} 合约规格 (回测使用)", expanded=False):
            cols = st.columns(6)
            with cols[0]:
                st.metric("品种", inst['name'])
            with cols[1]:
                st.metric("合约乘数", f"{inst['multiplier']}")
            with cols[2]:
                st.metric("最小变动", f"{inst['price_tick']}")
            with cols[3]:
                st.metric("保证金率", f"{inst['margin_rate']*100:.0f}%")
            with cols[4]:
                if inst['commission_fixed'] > 0:
                    st.metric("手续费", f"{inst['commission_fixed']}元/手")
                else:
                    st.metric("手续费率", f"{inst['commission_rate']*10000:.2f}%%")
            with cols[5]:
                st.metric("交易所", inst['exchange'])

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("总收益", f"¥{result.total_pnl:,.0f}", f"{result.total_return_pct:+.2f}%")
    with col2:
        st.metric("年化收益", f"{result.annual_return_pct:.2f}%")
    with col3:
        st.metric("最大回撤", f"{result.max_drawdown_pct:.2f}%", f"¥{result.max_drawdown_val:,.0f}")
    with col4:
        st.metric("夏普比率", f"{result.sharpe_ratio:.2f}")
    with col5:
        st.metric("胜率", f"{result.win_rate:.1f}%", f"{len([t for t in result.trades if t.pnl > 0])}/{len(result.trades)}")
    with col6:
        st.metric("盈亏比", f"{result.profit_factor:.2f}")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**收益指标**")
        st.write(f"初始资金: ¥{result.initial_capital:,.0f}")
        st.write(f"期末资金: ¥{result.final_capital:,.0f}")
        st.write(f"总盈亏: ¥{result.total_pnl:,.0f}")
        st.write(f"总收益率: {result.total_return_pct:.2f}%")
        st.write(f"年化收益: {result.annual_return_pct:.2f}%")
        st.write(f"总手续费: ¥{result.total_commission:,.0f}")

    with col2:
        st.write("**风险指标**")
        st.write(f"最大回撤: {result.max_drawdown_pct:.2f}%")
        st.write(f"回撤金额: ¥{result.max_drawdown_val:,.0f}")
        st.write(f"夏普比率: {result.sharpe_ratio:.2f}")
        st.write(f"索提诺比率: {result.sortino_ratio:.2f}")
        st.write(f"卡尔玛比率: {result.calmar_ratio:.2f}")

    with col3:
        st.write("**交易指标**")
        st.write(f"总交易数: {len(result.trades)}")
        st.write(f"胜率: {result.win_rate:.1f}%")
        st.write(f"盈亏比: {result.profit_factor:.2f}")
        st.write(f"平均盈利: ¥{result.avg_win:,.0f}")
        st.write(f"平均亏损: ¥{result.avg_loss:,.0f}")
        st.write(f"平均持仓: {result.avg_holding_days:.1f}天")


def render_backtest_equity_chart(result):
    """渲染资金曲线"""
    st.subheader("资金曲线")

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
            x=df['time'], y=df['equity'],
            name='账户净值',
            line=dict(color='#2196F3', width=2),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.1)'
        ),
        row=1, col=1
    )

    for trade in result.trades:
        color = '#4CAF50' if trade.pnl > 0 else '#F44336'
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_time],
                y=[trade.capital_before if trade.capital_before > 0 else result.initial_capital],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='#2196F3'),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[trade.exit_time], y=[trade.capital_after],
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color=color),
                showlegend=False
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=df['time'], y=-df['drawdown_pct'],
            name='回撤',
            line=dict(color='#F44336', width=1),
            fill='tozeroy',
            fillcolor='rgba(244, 67, 54, 0.3)'
        ),
        row=2, col=1
    )

    fig.update_layout(height=600, hovermode='x unified', showlegend=True)
    fig.update_yaxes(title_text="净值 (元)", row=1, col=1)
    fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_backtest_kline_with_trades(result, df_data):
    """渲染K线图并标记交易"""
    st.subheader("K线交易图")

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
        trade_options = [f"#{t.trade_id+1} {t.entry_time.strftime('%m-%d')}→{t.exit_time.strftime('%m-%d') if t.exit_time else ''} {'盈' if t.pnl > 0 else '亏'}{abs(t.pnl_pct):.1f}%"
                        for t in result.trades]
        selected_trade_idx = st.selectbox(
            "跳转到交易",
            options=range(len(trade_options)),
            format_func=lambda x: trade_options[x],
            key="kline_trade_select"
        )

    filtered_trades = [t for t in result.trades
                      if (('盈利' in result_filter and t.pnl > 0) or ('亏损' in result_filter and t.pnl <= 0))
                      and t.exit_tag in tag_filter]

    st.write(f"显示 **{len(filtered_trades)}** / {len(result.trades)} 笔交易")

    # 计算价格范围
    price_min = df_data['low'].min()
    price_max = df_data['high'].max()
    price_range = price_max - price_min
    y_min = price_min - price_range * 0.05
    y_max = price_max + price_range * 0.08

    # 创建K线图
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.15, 0.15],
        subplot_titles=('', '', '')
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
            increasing_line_color='#EF5350',
            decreasing_line_color='#26A69A',
            increasing_fillcolor='#EF5350',
            decreasing_fillcolor='#26A69A'
        ),
        row=1, col=1
    )

    # 成交量
    colors = ['#EF5350' if close >= open else '#26A69A'
              for close, open in zip(df_data['close'], df_data['open'])]
    fig.add_trace(
        go.Bar(x=df_data['time'], y=df_data['volume'], name='成交量', marker_color=colors, opacity=0.7),
        row=2, col=1
    )

    # 持仓盈亏曲线
    holding_pnl = []
    holding_time = []
    for t in filtered_trades:
        mask = (df_data['time'] >= t.entry_time) & (df_data['time'] <= t.exit_time)
        trade_data = df_data[mask]
        for _, row in trade_data.iterrows():
            pnl_pct = (row['close'] - t.entry_price) / t.entry_price * 100
            holding_pnl.append(pnl_pct)
            holding_time.append(row['time'])

    if holding_pnl:
        fig.add_trace(
            go.Scatter(
                x=holding_time, y=holding_pnl,
                mode='lines', name='持仓盈亏%',
                line=dict(color='#FF9800', width=1),
                fill='tozeroy', fillcolor='rgba(255, 152, 0, 0.2)'
            ),
            row=3, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    # 标记交易入场和出场
    for t in filtered_trades:
        is_win = t.pnl > 0

        # 入场标记
        entry_low = df_data[df_data['time'] == t.entry_time]['low'].values
        entry_y = entry_low[0] * 0.995 if len(entry_low) > 0 else t.entry_price

        fig.add_trace(
            go.Scatter(
                x=[t.entry_time], y=[entry_y],
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=16, color='#2196F3', line=dict(color='white', width=1)),
                text=[f'买{t.volume}手'],
                textposition='bottom center',
                textfont=dict(size=10, color='#2196F3'),
                name=f'入场#{t.trade_id+1}',
                showlegend=False,
                hovertemplate=f"<b>入场 #{t.trade_id+1}</b><br>时间: {t.entry_time.strftime('%Y-%m-%d')}<br>价格: {t.entry_price:.2f}<br>手数: {t.volume}<extra></extra>"
            ),
            row=1, col=1
        )

        # 出场标记
        if t.exit_time:
            exit_high = df_data[df_data['time'] == t.exit_time]['high'].values
            exit_y = exit_high[0] * 1.005 if len(exit_high) > 0 else t.exit_price
            exit_color = '#4CAF50' if is_win else '#F44336'

            fig.add_trace(
                go.Scatter(
                    x=[t.exit_time], y=[exit_y],
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', size=16, color=exit_color, line=dict(color='white', width=1)),
                    text=[f'{t.pnl_pct:+.1f}%'],
                    textposition='top center',
                    textfont=dict(size=10, color=exit_color, weight='bold'),
                    name=f'出场#{t.trade_id+1}',
                    showlegend=False,
                    hovertemplate=f"<b>出场 #{t.trade_id+1}</b><br>时间: {t.exit_time.strftime('%Y-%m-%d')}<br>价格: {t.exit_price:.2f}<br>盈亏: ¥{t.pnl:+,.0f} ({t.pnl_pct:+.2f}%)<br>原因: {t.exit_tag}<br>持仓: {t.holding_days}天<extra></extra>"
                ),
                row=1, col=1
            )

            # 连接线
            fig.add_trace(
                go.Scatter(
                    x=[t.entry_time, t.exit_time],
                    y=[t.entry_price, t.exit_price],
                    mode='lines',
                    line=dict(color=exit_color, width=2, dash='dot'),
                    opacity=0.6, showlegend=False, hoverinfo='skip'
                ),
                row=1, col=1
            )

            # 持仓区间背景色
            fig.add_shape(
                type="rect",
                x0=t.entry_time, x1=t.exit_time,
                y0=y_min, y1=y_max,
                fillcolor='rgba(76, 175, 80, 0.15)' if is_win else 'rgba(244, 67, 54, 0.15)',
                layer='below', line_width=0,
                row=1, col=1
            )

    # 聚焦到选中的交易
    if selected_trade_idx is not None and selected_trade_idx < len(result.trades):
        selected_trade = result.trades[selected_trade_idx]
        trade_start = selected_trade.entry_time
        trade_end = selected_trade.exit_time if selected_trade.exit_time else trade_start

        try:
            start_idx = df_data[df_data['time'] <= trade_start].index[-1] - 30
            end_idx = df_data[df_data['time'] >= trade_end].index[0] + 30
            start_idx = max(0, start_idx)
            end_idx = min(len(df_data) - 1, end_idx)

            x_start = df_data.iloc[start_idx]['time']
            x_end = df_data.iloc[end_idx]['time']

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

    fig.update_layout(
        height=700,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=30, b=30)
    )

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
            st.metric("盈亏金额", f"¥{t.pnl:+,.0f}", delta=f"{t.pnl_pct:+.2f}%")
            st.metric("出场原因", t.exit_tag)


def render_backtest_trades_table(result):
    """渲染交易列表"""
    st.subheader("交易记录")

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
    st.download_button("下载交易记录", csv, "trades.csv", "text/csv")


def render_backtest_statistics(result):
    """渲染统计分析"""
    st.subheader("统计分析")

    if not result.trades:
        st.warning("没有交易记录")
        return

    col1, col2 = st.columns(2)

    with col1:
        if result.exit_tag_stats is not None:
            st.write("**出场原因统计**")
            df_exit = result.exit_tag_stats.reset_index()
            df_exit.columns = ['出场原因', '次数', '总盈亏', '平均盈亏', '平均收益%']
            st.dataframe(df_exit, use_container_width=True, hide_index=True)

    with col2:
        st.write("**收益分布**")
        pnl_list = [t.pnl for t in result.trades]

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=pnl_list, nbinsx=20, marker_color='#2196F3'))
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(height=300, xaxis_title='盈亏金额 (元)', yaxis_title='次数', margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

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


def render_data_management_page():
    """渲染数据管理页面"""
    st.header("数据管理")

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
            if st.button("开始下载", type="primary", use_container_width=True):
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
        st.info("分钟数据来自新浪财经，约有最近1000根K线")

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
            if st.button("开始下载分钟数据", type="primary"):
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
        if st.button("刷新"):
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


# 页面配置
st.set_page_config(
    page_title="期货量化交易系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    /* 主题色 - 使用深色文字 */
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ecc71;
        --danger-color: #e74c3c;
        --warning-color: #f39c12;
        --bg-dark: #1e1e1e;
        --bg-card: #2d2d2d;
        --text-primary: #000000;
        --text-secondary: #333333;
    }

    /* 隐藏默认菜单 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 全局文字颜色 */
    .stMarkdown, .stText, p, span, label, div {
        color: #000000 !important;
    }

    /* 卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #dee2e6;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #000000;
    }

    .metric-label {
        font-size: 14px;
        color: #000000;
        margin-bottom: 5px;
    }

    .metric-change-positive {
        color: #2ecc71;
        font-size: 14px;
    }

    .metric-change-negative {
        color: #e74c3c;
        font-size: 14px;
    }

    /* 状态指示器 */
    .status-running {
        color: #2ecc71 !important;
        font-weight: bold;
    }

    .status-stopped {
        color: #e74c3c !important;
        font-weight: bold;
    }

    /* 表格样式优化 */
    .dataframe {
        font-size: 13px !important;
        color: #000000 !important;
    }

    /* 侧边栏文字 */
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    /* 标题 */
    h1 {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    h2, h3 {
        color: #000000 !important;
    }

    /* 标签和说明文字 */
    .stSelectbox label, .stMultiSelect label, .stNumberInput label,
    .stDateInput label, .stTextInput label, .stCheckbox label {
        color: #000000 !important;
    }

    /* Expander 标题 */
    .streamlit-expanderHeader {
        color: #000000 !important;
    }

    /* Tab 标签 */
    .stTabs [data-baseweb="tab"] {
        color: #000000 !important;
    }

    /* Metric 组件 */
    [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }

    /* Caption 说明文字 */
    .stCaption {
        color: #333333 !important;
    }

    /* 按钮 */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 8px 16px;
    }

    .stButton > button:hover {
        background-color: #1668a3;
    }

    /* 盈亏颜色 */
    .profit {
        color: #2ecc71 !important;
    }

    .loss {
        color: #e74c3c !important;
    }
</style>
""", unsafe_allow_html=True)


def render_metric_card(label: str, value: str, change: str = None, change_type: str = "neutral"):
    """渲染指标卡片"""
    change_class = "metric-change-positive" if change_type == "positive" else "metric-change-negative"
    change_html = f'<div class="{change_class}">{change}</div>' if change else ""

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)


def main():
    """主函数"""
    # 侧边栏
    with st.sidebar:
        st.title("📈 期货量化系统")
        st.markdown("---")

        # 导航 - 6个一级菜单
        page = st.radio(
            "功能模块",
            ["仪表盘", "模拟交易", "实盘交易", "风控中心", "回测系统", "系统设置"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # 系统状态
        st.markdown("### 系统状态")

        # 这里应该从实际引擎获取状态
        engine_running = st.session_state.get('engine_running', False)

        if engine_running:
            st.markdown('<span class="status-running">● 运行中</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-stopped">● 已停止</span>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("启动", use_container_width=True, disabled=engine_running):
                st.session_state['engine_running'] = True
                st.rerun()
        with col2:
            if st.button("停止", use_container_width=True, disabled=not engine_running):
                st.session_state['engine_running'] = False
                st.rerun()

        st.markdown("---")
        st.caption(f"更新时间: {datetime.now().strftime('%H:%M:%S')}")

    # 主内容区 - 6个页面
    if page == "仪表盘":
        render_dashboard()
    elif page == "模拟交易":
        if HAS_SIM_TRADING:
            render_sim_trading_page()
        else:
            st.error("模拟交易模块未加载，请检查依赖")
    elif page == "实盘交易":
        if HAS_LIVE_TRADING:
            render_live_trading_page()
        else:
            st.error("实盘交易模块未加载，请检查依赖")
    elif page == "风控中心":
        render_risk_center()
    elif page == "回测系统":
        render_backtest()
    elif page == "系统设置":
        render_settings()


def render_dashboard():
    """渲染仪表盘 - 系统概览"""
    st.title("系统概览")

    # 获取引擎状态
    sim_engine = st.session_state.get('sim_engine')
    live_engine = st.session_state.get('live_engine')

    sim_running = sim_engine is not None and sim_engine.is_running if sim_engine else False
    live_running = live_engine is not None and live_engine.is_running if live_engine else False

    # 系统状态卡片
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("模拟交易")
        if sim_running:
            st.success("运行中")
            account = sim_engine.get_account()
            if account:
                st.metric("账户权益", f"¥{account.balance:,.0f}")
                st.metric("持仓数量", f"{len(sim_engine.get_positions())}")
        else:
            st.info("未启动")
            st.caption("前往「模拟交易」页面启动")

    with col2:
        st.subheader("实盘交易")
        if live_running:
            st.success("运行中")
            account = live_engine.get_account()
            if account:
                st.metric("账户权益", f"¥{account.balance:,.0f}")
                st.metric("持仓数量", f"{len(live_engine.get_positions())}")
        else:
            st.warning("未启动")
            st.caption("前往「实盘交易」页面启动")

    with col3:
        st.subheader("系统信息")
        st.metric("已配置策略", f"{len(get_all_strategies())}")
        st.metric("已配置品种", f"{len(INSTRUMENTS)}")

    st.markdown("---")

    # 快速操作
    st.subheader("快速入口")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**模拟交易**")
        st.caption("使用真实行情数据进行策略验证")
        if st.button("进入模拟交易", use_container_width=True):
            st.session_state.nav_page = "模拟交易"
            st.rerun()

    with col2:
        st.markdown("**策略回测**")
        st.caption("历史数据回测，评估策略表现")
        if st.button("进入回测系统", use_container_width=True):
            st.session_state.nav_page = "回测系统"
            st.rerun()

    with col3:
        st.markdown("**风控中心**")
        st.caption("设置风控规则，监控交易风险")
        if st.button("进入风控中心", use_container_width=True):
            st.session_state.nav_page = "风控中心"
            st.rerun()

    with col4:
        st.markdown("**系统设置**")
        st.caption("配置天勤账号、品种参数等")
        if st.button("进入系统设置", use_container_width=True):
            st.session_state.nav_page = "系统设置"
            st.rerun()

    st.markdown("---")

    # 使用说明
    st.subheader("使用流程")
    st.markdown("""
    1. **回测验证** → 在「回测系统」中测试策略，确认参数
    2. **模拟交易** → 在「模拟交易」中使用真实行情验证策略
    3. **实盘上线** → 确认无误后，在「实盘交易」中启动真实交易
    """)


def render_strategy_management():
    """渲染策略管理页面"""
    st.title("策略管理")

    # 策略列表
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("已加载策略")

    with col2:
        if st.button("+ 添加策略", use_container_width=True):
            st.session_state['show_add_strategy'] = True

    # 策略表格
    strategies_df = pd.DataFrame({
        '策略名称': ['WaveTrend趋势', 'MACD动量', 'EMA突破'],
        '状态': ['运行中', '运行中', '已停止'],
        '交易品种': ['RB, I, AU', 'CU, AL', 'RB'],
        '今日盈亏': ['+2,350', '+850', '0'],
        '累计盈亏': ['+25,680', '+8,450', '+3,200'],
        '胜率': ['58.3%', '52.1%', '61.5%'],
        '最大回撤': ['8.5%', '12.3%', '6.8%']
    })

    # 显示策略列表
    for idx, row in strategies_df.iterrows():
        with st.expander(f"📊 {row['策略名称']} - {row['状态']}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("今日盈亏", row['今日盈亏'])
            with col2:
                st.metric("累计盈亏", row['累计盈亏'])
            with col3:
                st.metric("胜率", row['胜率'])
            with col4:
                st.metric("最大回撤", row['最大回撤'])

            st.write(f"**交易品种**: {row['交易品种']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.button("编辑参数", key=f"edit_{idx}")
            with col2:
                if row['状态'] == '运行中':
                    st.button("暂停", key=f"pause_{idx}")
                else:
                    st.button("启动", key=f"start_{idx}")
            with col3:
                st.button("移除", key=f"remove_{idx}")

    # 添加策略弹窗
    if st.session_state.get('show_add_strategy', False):
        st.markdown("---")
        st.subheader("添加新策略")

        col1, col2 = st.columns(2)

        with col1:
            strategy_type = st.selectbox(
                "选择策略",
                ["WaveTrend趋势策略", "MACD动量策略", "EMA突破策略", "自定义策略"]
            )

            symbols = st.multiselect(
                "交易品种",
                ["RB", "I", "AU", "CU", "AL", "NI", "TA", "MA", "PP"]
            )

        with col2:
            st.write("**策略参数**")
            param1 = st.number_input("参数1", value=10)
            param2 = st.number_input("参数2", value=20)
            param3 = st.number_input("参数3", value=50)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("确认添加", use_container_width=True):
                st.success("策略添加成功!")
                st.session_state['show_add_strategy'] = False
                st.rerun()
        with col2:
            if st.button("取消", use_container_width=True):
                st.session_state['show_add_strategy'] = False
                st.rerun()


def render_position_monitor():
    """渲染持仓监控页面"""
    st.title("持仓监控")

    # 汇总信息
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("总持仓市值", "¥358,000")
    with col2:
        st.metric("浮动盈亏", "¥1,950", "+0.55%")
    with col3:
        st.metric("已用保证金", "¥27,230")
    with col4:
        st.metric("保证金占用", "21.7%")

    st.markdown("---")

    # 持仓明细
    st.subheader("持仓明细")

    positions_df = pd.DataFrame({
        '合约': ['RB2505', 'I2505', 'AU2506', 'CU2505'],
        '方向': ['多', '多', '空', '多'],
        '数量': [5, 3, 2, 1],
        '开仓价': [3580.0, 820.0, 580.0, 75200.0],
        '现价': [3620.0, 815.0, 575.0, 75500.0],
        '浮盈(元)': [2000.0, -150.0, 100.0, 300.0],
        '浮盈%': [1.12, -0.61, 0.86, 0.40],
        '保证金': [17900.0, 2460.0, 1160.0, 7520.0],
        '持仓时间': ['2小时', '1天', '3小时', '2天'],
        '策略': ['WaveTrend', 'WaveTrend', 'MACD', 'EMA']
    })

    # 格式化显示
    def color_pnl(val):
        if isinstance(val, (int, float)):
            color = '#2ecc71' if val > 0 else '#e74c3c' if val < 0 else '#ffffff'
            return f'color: {color}'
        return ''

    styled_df = positions_df.style.applymap(
        color_pnl, subset=['浮盈(元)', '浮盈%']
    )

    st.dataframe(styled_df, hide_index=True, use_container_width=True, height=300)

    st.markdown("---")

    # 持仓操作
    st.subheader("快捷操作")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("一键平多", use_container_width=True):
            st.warning("确认平掉所有多头持仓?")
    with col2:
        if st.button("一键平空", use_container_width=True):
            st.warning("确认平掉所有空头持仓?")
    with col3:
        if st.button("全部平仓", use_container_width=True, type="primary"):
            st.error("确认平掉所有持仓?")
    with col4:
        if st.button("刷新数据", use_container_width=True):
            st.rerun()


def render_order_management():
    """渲染订单管理页面"""
    st.title("订单管理")

    # 选项卡
    tab1, tab2, tab3 = st.tabs(["活动订单", "今日成交", "历史订单"])

    with tab1:
        st.subheader("活动订单")

        active_orders = pd.DataFrame({
            '订单号': ['ORD001', 'ORD002'],
            '时间': ['14:35:20', '14:20:15'],
            '合约': ['RB2505', 'I2505'],
            '方向': ['买', '卖'],
            '开平': ['开', '平'],
            '报价': [3575, 825],
            '数量': [2, 1],
            '已成': [0, 0],
            '状态': ['等待成交', '等待成交'],
            '策略': ['WaveTrend', 'MACD']
        })

        st.dataframe(active_orders, hide_index=True, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("撤销选中", use_container_width=True):
                st.info("请先选择要撤销的订单")
        with col2:
            if st.button("全部撤单", use_container_width=True):
                st.warning("确认撤销所有挂单?")

    with tab2:
        st.subheader("今日成交")

        trades_df = pd.DataFrame({
            '成交号': ['TRD001', 'TRD002', 'TRD003', 'TRD004'],
            '时间': ['14:35:20', '14:20:15', '11:30:00', '10:45:30'],
            '合约': ['RB2505', 'I2505', 'AU2506', 'RB2505'],
            '方向': ['买', '买', '卖', '买'],
            '开平': ['开', '开', '开', '开'],
            '价格': [3580, 820, 580, 3570],
            '数量': [2, 3, 2, 3],
            '手续费': [8.5, 6.0, 12.0, 8.5],
            '策略': ['WaveTrend', 'WaveTrend', 'MACD', 'WaveTrend']
        })

        st.dataframe(trades_df, hide_index=True, use_container_width=True)

        # 成交统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("成交笔数", "4")
        with col2:
            st.metric("成交手数", "10")
        with col3:
            st.metric("手续费合计", "¥35.0")
        with col4:
            st.metric("已实现盈亏", "¥2,350")

    with tab3:
        st.subheader("历史订单查询")

        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input("开始日期", datetime.now() - timedelta(days=7))
        with col2:
            end_date = st.date_input("结束日期", datetime.now())
        with col3:
            symbol_filter = st.selectbox("品种", ["全部", "RB", "I", "AU", "CU"])

        if st.button("查询"):
            st.info("查询历史订单...")


def render_risk_center():
    """渲染风控中心页面"""
    st.title("风控中心")

    # 风险状态
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        risk_level = "low"  # 从引擎获取

        if risk_level == "low":
            st.success("### 风险状态: 安全")
        elif risk_level == "medium":
            st.warning("### 风险状态: 警告")
        elif risk_level == "high":
            st.warning("### 风险状态: 高风险")
        else:
            st.error("### 风险状态: 危险")

    st.markdown("---")

    # 风险指标
    st.subheader("风险指标")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("保证金占用", "21.7%", "限制: 80%")
    with col2:
        st.metric("日亏损", "-0.5%", "限制: 5%")
    with col3:
        st.metric("最大回撤", "3.2%", "限制: 15%")
    with col4:
        st.metric("连续亏损", "1次", "限制: 5次")

    st.markdown("---")

    # 风控设置
    st.subheader("风控设置")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**持仓限制**")
        max_pos_per_symbol = st.number_input("单品种最大持仓", value=10, min_value=1)
        max_pos_total = st.number_input("总最大持仓", value=50, min_value=1)

        st.write("**资金风控**")
        max_margin_ratio = st.slider("最大保证金占用比例", 0.0, 1.0, 0.8)
        min_available = st.number_input("最小可用资金", value=10000, min_value=0)

    with col2:
        st.write("**亏损控制**")
        max_daily_loss = st.slider("日最大亏损比例", 0.0, 0.2, 0.05)
        max_drawdown = st.slider("最大回撤比例", 0.0, 0.3, 0.15)
        max_consecutive = st.number_input("最大连续亏损次数", value=5, min_value=1)

        st.write("**其他设置**")
        force_close = st.checkbox("达到限制时强制平仓", value=True)
        allow_open = st.checkbox("高风险时允许开仓", value=False)

    if st.button("保存设置", use_container_width=True):
        st.success("风控设置已保存!")

    st.markdown("---")

    # 风控日志
    st.subheader("风控日志")

    logs_df = pd.DataFrame({
        '时间': ['14:35:20', '14:20:15', '11:30:00'],
        '级别': ['INFO', 'WARNING', 'INFO'],
        '消息': [
            '订单风控检查通过: RB2505 买开2手',
            '日亏损接近限制: -4.2% (限制: -5%)',
            '新策略加入: WaveTrend'
        ]
    })

    st.dataframe(logs_df, hide_index=True, use_container_width=True)


def render_backtest():
    """渲染回测系统页面 - 完整版"""
    st.title("回测系统")

    # 回测子页面选择
    backtest_page = st.radio(
        "功能选择",
        ["策略回测", "数据管理"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    if backtest_page == "数据管理":
        render_data_management_page()
    else:
        # 渲染回测配置
        config = render_backtest_config()

        if config is None:
            return

        result_container = st.container()

        # 运行回测
        if config['run_backtest']:
            run_backtest_and_display(config, result_container)

        # 显示回测结果
        if 'backtest_result' in st.session_state:
            result = st.session_state['backtest_result']
            df_data = st.session_state.get('backtest_df_data', None)

            with result_container:
                tabs = st.tabs(["概览", "K线交易图", "资金曲线", "交易记录", "统计分析"])

                with tabs[0]:
                    render_backtest_overview(result)

                with tabs[1]:
                    render_backtest_kline_with_trades(result, df_data)

                with tabs[2]:
                    render_backtest_equity_chart(result)

                with tabs[3]:
                    render_backtest_trades_table(result)

                with tabs[4]:
                    render_backtest_statistics(result)


def render_settings():
    """渲染系统设置页面"""
    st.title("系统设置")

    tab1, tab2, tab3, tab4 = st.tabs(["基础设置", "品种配置", "网关设置", "数据管理"])

    with tab1:
        st.subheader("基础设置")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**账户设置**")
            st.number_input("初始资金", value=100000, min_value=10000)
            st.selectbox("结算货币", ["CNY", "USD"])

            st.write("**显示设置**")
            st.checkbox("深色模式", value=True)
            st.selectbox("刷新频率", ["1秒", "3秒", "5秒", "10秒"])

        with col2:
            st.write("**通知设置**")
            st.checkbox("成交通知", value=True)
            st.checkbox("风控预警通知", value=True)
            st.checkbox("策略信号通知", value=False)

            st.write("**日志设置**")
            st.selectbox("日志级别", ["DEBUG", "INFO", "WARNING", "ERROR"])
            st.checkbox("保存日志到文件", value=True)

    with tab2:
        st.subheader("品种配置")

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
                st.text_input("品种代码")
                st.text_input("品种名称")
                st.number_input("合约乘数", value=10)
            with col2:
                st.number_input("保证金率", value=0.1, format="%.2f")
                st.selectbox("手续费类型", ["按比例", "固定金额"])
                st.number_input("手续费", value=0.0001, format="%.4f")

            st.button("保存品种配置")

    with tab3:
        st.subheader("TqSdk连接设置")

        # 加载配置
        tq_config = load_tq_config_for_settings()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**天勤账号**")
            tq_user = st.text_input("天勤用户名", value=tq_config.get('tq_user', ''), key="settings_tq_user")
            tq_password = st.text_input("天勤密码", type="password", value=tq_config.get('tq_password', ''), key="settings_tq_password")

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
                broker_id = st.text_input("期货公司代码", value=tq_config.get('broker_id', ''), key="settings_broker_id")
                td_account = st.text_input("交易账号", value=tq_config.get('td_account', ''), key="settings_td_account")
                td_password = st.text_input("交易密码", type="password", value=tq_config.get('td_password', ''), key="settings_td_password")
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
                test_tq_connection_settings(tq_user, tq_password)

    with tab4:
        st.subheader("数据管理")

        st.write("**数据库信息**")
        st.info("数据来源: TianQin量化数据库")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("K线数据量", "12.5GB")
            st.metric("交易记录数", "15,680")

        with col2:
            st.metric("数据时间范围", "2020-01 至 2025-12")
            st.metric("品种数量", "45")

        st.write("**数据操作**")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("更新数据", use_container_width=True):
                st.info("正在更新数据...")
        with col2:
            if st.button("清理缓存", use_container_width=True):
                st.success("缓存已清理")
        with col3:
            if st.button("备份数据库", use_container_width=True):
                st.info("正在备份...")


if __name__ == "__main__":
    main()
