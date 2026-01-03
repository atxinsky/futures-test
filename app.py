# coding=utf-8
"""
期货策略回测系统
参考banbot设计的专业回测可视化界面
支持多策略选择和动态参数配置
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
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """加载数据文件"""
    df = pd.read_csv(file_path)
    # 尝试自动识别列名
    if len(df.columns) >= 5:
        df.columns = ['time', 'open', 'high', 'low', 'close'] + list(df.columns[5:])
    df['time'] = pd.to_datetime(df['time'])
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
        if any(k in p.name for k in ['len', 'period', 'ma', 'ema', 'sma', 'fast', 'slow', 'bb']):
            grouped_params['均线/周期参数'].append(p)
        elif any(k in p.name for k in ['stop', 'atr', 'risk', 'adx']):
            grouped_params['风控参数'].append(p)
        elif any(k in p.name for k in ['capital', 'risk_rate', 'position']):
            grouped_params['仓位参数'].append(p)
        else:
            grouped_params['其他参数'].append(p)

    # 渲染各组参数
    for group_name, group_params in grouped_params.items():
        if not group_params:
            continue

        with st.sidebar.expander(group_name, expanded=True):
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
                        float(p.step) if p.step else 0.1,
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


def render_sidebar():
    """渲染侧边栏配置"""
    st.sidebar.title("⚙️ 回测配置")

    # ========== 策略选择 ==========
    st.sidebar.subheader("🎯 策略选择")

    strategies = get_all_strategies()
    strategy_names = list(strategies.keys())
    strategy_display = {k: v.display_name for k, v in strategies.items()}

    selected_strategy_name = st.sidebar.selectbox(
        "选择策略",
        options=strategy_names,
        format_func=lambda x: f"{strategy_display[x]} ({x})"
    )

    strategy_class = strategies[selected_strategy_name]

    # 显示策略信息
    with st.sidebar.expander("📖 策略说明", expanded=False):
        st.markdown(f"**{strategy_class.display_name}**")
        st.markdown(f"*版本: {strategy_class.version} | 作者: {strategy_class.author}*")
        st.markdown(strategy_class.description)

    # 导入外部策略
    st.sidebar.markdown("---")
    with st.sidebar.expander("📥 导入外部策略", expanded=False):
        uploaded_file = st.file_uploader(
            "上传策略文件 (.py)",
            type=['py'],
            help="上传继承自BaseStrategy的策略Python文件"
        )
        if uploaded_file is not None:
            # 保存到临时目录
            temp_path = os.path.join(os.path.dirname(__file__), 'strategies', f'_temp_{uploaded_file.name}')
            try:
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                # 加载策略
                new_strategy = load_strategy_from_file(temp_path)
                st.success(f"✅ 成功导入策略: {new_strategy.display_name}")
                # 刷新页面以显示新策略
                st.rerun()
            except Exception as e:
                st.error(f"导入失败: {e}")
            finally:
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        strategy_file_path = st.text_input(
            "或输入策略文件路径",
            placeholder="D:/my_strategies/my_strategy.py"
        )
        if st.button("加载策略") and strategy_file_path:
            try:
                new_strategy = load_strategy_from_file(strategy_file_path)
                st.success(f"✅ 成功导入策略: {new_strategy.display_name}")
                st.rerun()
            except Exception as e:
                st.error(f"加载失败: {e}")

    st.sidebar.markdown("---")

    # ========== 品种选择 ==========
    st.sidebar.subheader("📌 品种设置")

    symbol = st.sidebar.selectbox(
        "选择品种",
        options=list(INSTRUMENTS.keys()),
        format_func=lambda x: f"{x} - {INSTRUMENTS[x]['name']}"
    )

    inst = get_instrument(symbol)

    # 显示品种信息
    with st.sidebar.expander("品种详情", expanded=False):
        st.write(f"**交易所**: {inst['exchange']}")
        st.write(f"**合约乘数**: {inst['multiplier']} 元/点")
        st.write(f"**最小变动**: {inst['price_tick']}")
        st.write(f"**保证金率**: {inst['margin_rate']*100:.1f}%")
        if inst['commission_fixed'] > 0:
            st.write(f"**手续费**: {inst['commission_fixed']} 元/手")
        else:
            st.write(f"**手续费率**: {inst['commission_rate']*10000:.2f}‱")
        st.write(f"**夜盘**: {'是' if inst['night_trade'] else '否'}")

    # ========== 数据文件 ==========
    st.sidebar.subheader("📁 数据文件")
    data_dir = st.sidebar.text_input("数据目录", value="D:/期货/股指期货")

    # 扫描目录中的CSV文件
    csv_files = []
    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if csv_files:
        data_file = st.sidebar.selectbox("选择数据文件", options=csv_files)
        file_path = os.path.join(data_dir, data_file)
    else:
        file_path = st.sidebar.text_input("数据文件路径")

    # ========== 策略参数 ==========
    st.sidebar.subheader("🔧 策略参数")
    params = render_strategy_params(strategy_class)

    # ========== 资金设置 ==========
    st.sidebar.subheader("💰 资金设置")
    initial_capital = st.sidebar.number_input(
        "初始资金 (元)",
        min_value=100000,
        max_value=100000000,
        value=1000000,
        step=100000
    )

    return symbol, file_path, params, initial_capital, strategy_class


def render_overview(result):
    """渲染概览页"""
    st.header("📊 回测概览")

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
        st.subheader("💰 收益指标")
        st.write(f"**初始资金**: ¥{result.initial_capital:,.0f}")
        st.write(f"**期末资金**: ¥{result.final_capital:,.0f}")
        st.write(f"**总盈亏**: ¥{result.total_pnl:,.0f}")
        st.write(f"**总收益率**: {result.total_return_pct:.2f}%")
        st.write(f"**年化收益**: {result.annual_return_pct:.2f}%")
        st.write(f"**总手续费**: ¥{result.total_commission:,.0f}")

    with col2:
        st.subheader("📉 风险指标")
        st.write(f"**最大回撤**: {result.max_drawdown_pct:.2f}%")
        st.write(f"**回撤金额**: ¥{result.max_drawdown_val:,.0f}")
        st.write(f"**夏普比率**: {result.sharpe_ratio:.2f}")
        st.write(f"**索提诺比率**: {result.sortino_ratio:.2f}")
        st.write(f"**卡尔玛比率**: {result.calmar_ratio:.2f}")
        st.write(f"**收益/回撤**: {result.total_return_pct / result.max_drawdown_pct:.2f}" if result.max_drawdown_pct > 0 else "**收益/回撤**: N/A")

    with col3:
        st.subheader("📈 交易指标")
        st.write(f"**总交易数**: {len(result.trades)}")
        st.write(f"**胜率**: {result.win_rate:.1f}%")
        st.write(f"**盈亏比**: {result.profit_factor:.2f}")
        st.write(f"**平均盈利**: ¥{result.avg_win:,.0f}")
        st.write(f"**平均亏损**: ¥{result.avg_loss:,.0f}")
        st.write(f"**平均持仓**: {result.avg_holding_days:.1f}天")


def render_equity_chart(result):
    """渲染资金曲线"""
    st.header("💹 资金曲线")

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
    st.header("📋 交易记录")

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
            '盈亏点': f"{(t.exit_price - t.entry_price) * t.direction:.1f}" if t.exit_price else '',
            '盈亏%': f"{t.pnl_pct:+.2f}%",
            '盈亏额': f"¥{t.pnl:+,.0f}",
            '手续费': f"¥{t.commission:.0f}",
            '出场原因': t.exit_tag,
            '结果': '盈' if t.pnl > 0 else '亏'
        })

    df_trades = pd.DataFrame(trades_data)

    # 筛选器
    col1, col2, col3 = st.columns(3)
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
    with col3:
        sort_by = st.selectbox(
            "排序",
            options=['编号', '盈亏额', '持仓(天)', '入场时间']
        )

    # 应用筛选
    df_filtered = df_trades[
        (df_trades['结果'].isin(result_filter)) &
        (df_trades['出场原因'].isin(tag_filter))
    ]

    # 排序
    if sort_by == '盈亏额':
        df_filtered['_sort'] = df_filtered['盈亏额'].str.replace('[¥,]', '', regex=True).astype(float)
        df_filtered = df_filtered.sort_values('_sort', ascending=False).drop('_sort', axis=1)

    st.dataframe(df_filtered, use_container_width=True, hide_index=True)

    # 下载按钮
    csv = df_filtered.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        "📥 下载交易记录",
        csv,
        "trades.csv",
        "text/csv"
    )


def render_kline_analysis(result, df_data, params, strategy_class):
    """渲染K线分析"""
    st.header("📈 K线分析")

    if not result.trades:
        st.warning("没有交易记录")
        return

    # 交易选择器
    trade_options = []
    for t in result.trades:
        status = "盈" if t.pnl > 0 else "亏"
        trade_options.append(
            f"[{status}] #{t.trade_id+1} | {t.entry_time.strftime('%Y-%m-%d')} → "
            f"{t.exit_time.strftime('%Y-%m-%d')} | 盈亏: ¥{t.pnl:+,.0f}"
        )

    selected_idx = st.selectbox(
        "选择交易",
        range(len(trade_options)),
        format_func=lambda x: trade_options[x]
    )

    trade = result.trades[selected_idx]

    # K线图
    start_date = trade.entry_time - timedelta(days=60)
    end_date = trade.exit_time + timedelta(days=30)

    mask = (df_data['time'] >= start_date) & (df_data['time'] <= end_date)
    df_plot = df_data[mask].copy()

    # 使用策略计算指标
    strategy_instance = strategy_class(params)
    df_plot = strategy_instance.calculate_indicators(df_plot)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        subplot_titles=('K线图', '成交量')
    )

    # K线
    fig.add_trace(
        go.Candlestick(
            x=df_plot['time'],
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='K线',
            increasing_line_color='#F44336',
            decreasing_line_color='#4CAF50'
        ),
        row=1, col=1
    )

    # 尝试添加均线指标
    indicator_cols = ['ema_short', 'ema_long', 'ma_fast', 'ma_slow', 'bb_upper', 'bb_middle', 'bb_lower',
                      'high_line', 'entry_high', 'exit_low']
    colors = ['orange', 'blue', 'orange', 'blue', 'gray', 'purple', 'gray', 'purple', 'green', 'red']

    for col, color in zip(indicator_cols, colors):
        if col in df_plot.columns:
            fig.add_trace(
                go.Scatter(x=df_plot['time'], y=df_plot[col],
                           name=col, line=dict(color=color, width=1)),
                row=1, col=1
            )

    # 买入标记
    fig.add_trace(
        go.Scatter(
            x=[trade.entry_time],
            y=[trade.entry_price],
            mode='markers+text',
            marker=dict(symbol='triangle-up', size=20, color='#2196F3'),
            text=[f"买入 {trade.entry_price:.1f}"],
            textposition='bottom center',
            textfont=dict(size=12, color='#2196F3'),
            showlegend=False
        ),
        row=1, col=1
    )

    # 卖出标记
    exit_color = '#4CAF50' if trade.pnl > 0 else '#F44336'
    fig.add_trace(
        go.Scatter(
            x=[trade.exit_time],
            y=[trade.exit_price],
            mode='markers+text',
            marker=dict(symbol='triangle-down', size=20, color=exit_color),
            text=[f"卖出 {trade.exit_price:.1f}"],
            textposition='top center',
            textfont=dict(size=12, color=exit_color),
            showlegend=False
        ),
        row=1, col=1
    )

    # 连接线
    fig.add_trace(
        go.Scatter(
            x=[trade.entry_time, trade.exit_time],
            y=[trade.entry_price, trade.exit_price],
            mode='lines',
            line=dict(color=exit_color, width=2, dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )

    # 成交量
    if 'volume' in df_plot.columns:
        colors = ['#F44336' if c >= o else '#4CAF50' for c, o in zip(df_plot['close'], df_plot['open'])]
        fig.add_trace(
            go.Bar(x=df_plot['time'], y=df_plot['volume'], name='成交量', marker_color=colors),
            row=2, col=1
        )

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 交易详情
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("入场信息")
        st.write(f"**时间**: {trade.entry_time.strftime('%Y-%m-%d')}")
        st.write(f"**价格**: {trade.entry_price:.2f}")
        st.write(f"**手数**: {trade.volume}")
        st.write(f"**信号**: {trade.entry_tag}")

    with col2:
        st.subheader("出场信息")
        st.write(f"**时间**: {trade.exit_time.strftime('%Y-%m-%d')}")
        st.write(f"**价格**: {trade.exit_price:.2f}")
        st.write(f"**持仓**: {trade.holding_days}天")
        st.write(f"**原因**: {trade.exit_tag}")

    # 盈亏
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pnl_points = (trade.exit_price - trade.entry_price) * trade.direction
        st.metric("盈亏点数", f"{pnl_points:+.1f}")
    with col2:
        st.metric("盈亏比例", f"{trade.pnl_pct:+.2f}%")
    with col3:
        st.metric("盈亏金额", f"¥{trade.pnl:+,.0f}")
    with col4:
        st.metric("手续费", f"¥{trade.commission:.0f}")


def render_statistics(result):
    """渲染统计分析"""
    st.header("📊 统计分析")

    tab1, tab2, tab3 = st.tabs(["按时间", "按出场原因", "收益分布"])

    with tab1:
        if result.yearly_stats is not None:
            st.subheader("年度统计")
            st.dataframe(result.yearly_stats.round(2), use_container_width=True)

        if result.monthly_stats is not None:
            st.subheader("月度统计")
            # 月度收益热力图
            df_monthly = result.monthly_stats.reset_index()
            df_monthly['year'] = df_monthly['exit_month'].dt.year
            df_monthly['month'] = df_monthly['exit_month'].dt.month

            pivot = df_monthly.pivot(index='year', columns='month', values='pnl')

            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=[f'{m}月' for m in pivot.columns],
                y=pivot.index,
                colorscale='RdYlGn',
                text=[[f'¥{v:,.0f}' if not pd.isna(v) else '' for v in row] for row in pivot.values],
                texttemplate='%{text}',
                hovertemplate='%{y}年%{x}: %{text}<extra></extra>'
            ))
            fig.update_layout(title='月度收益热力图', height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if result.exit_tag_stats is not None:
            st.subheader("出场原因统计")
            df_exit = result.exit_tag_stats.reset_index()
            df_exit.columns = ['出场原因', '次数', '总盈亏', '平均盈亏', '平均收益%']
            st.dataframe(df_exit, use_container_width=True, hide_index=True)

            # 饼图
            fig = go.Figure(data=[go.Pie(
                labels=df_exit['出场原因'],
                values=df_exit['次数'],
                hole=0.4
            )])
            fig.update_layout(title='出场原因分布')
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if result.trades:
            st.subheader("收益分布")
            pnl_list = [t.pnl for t in result.trades]

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=pnl_list,
                nbinsx=20,
                marker_color='#2196F3'
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(
                title='单笔交易盈亏分布',
                xaxis_title='盈亏金额 (元)',
                yaxis_title='次数'
            )
            st.plotly_chart(fig, use_container_width=True)

            # 盈亏对比
            wins = [t.pnl for t in result.trades if t.pnl > 0]
            losses = [t.pnl for t in result.trades if t.pnl <= 0]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("盈利交易", f"{len(wins)}笔", f"总计 ¥{sum(wins):,.0f}")
                st.metric("最大单笔盈利", f"¥{max(wins):,.0f}" if wins else "¥0")
                st.metric("平均盈利", f"¥{np.mean(wins):,.0f}" if wins else "¥0")

            with col2:
                st.metric("亏损交易", f"{len(losses)}笔", f"总计 ¥{sum(losses):,.0f}")
                st.metric("最大单笔亏损", f"¥{min(losses):,.0f}" if losses else "¥0")
                st.metric("平均亏损", f"¥{np.mean(losses):,.0f}" if losses else "¥0")


def main():
    st.title("📊 期货策略回测系统")
    st.markdown("*支持多策略选择和动态参数配置*")

    # 侧边栏配置
    symbol, file_path, params, initial_capital, strategy_class = render_sidebar()

    # 检查文件
    if not file_path or not os.path.exists(file_path):
        st.warning("请在侧边栏选择数据文件")
        st.info("""
        **数据格式要求:**
        - CSV文件，包含列: time, open, high, low, close, volume (可选)
        - time格式: YYYY-MM-DD 或 YYYY/MM/DD
        """)

        # 显示已加载策略
        st.subheader("📋 已加载策略")
        strategies = list_strategies()
        for s in strategies:
            with st.expander(f"**{s['display_name']}** ({s['name']})"):
                st.write(f"*版本: {s['version']} | 作者: {s['author']}*")
                st.markdown(s['description'])
                st.write("**参数列表:**")
                for p in s['params']:
                    st.write(f"- {p['label']} ({p['name']}): 默认={p['default']}, 范围=[{p['min_val']}, {p['max_val']}]")
        return

    # 加载数据
    try:
        df_data = load_data(file_path)
        st.sidebar.success(f"✅ 已加载 {len(df_data)} 条数据")
        st.sidebar.caption(f"{df_data['time'].min().strftime('%Y-%m-%d')} ~ {df_data['time'].max().strftime('%Y-%m-%d')}")
    except Exception as e:
        st.error(f"加载数据失败: {e}")
        return

    # 运行回测按钮
    if st.sidebar.button("🚀 运行回测", type="primary", use_container_width=True):
        with st.spinner(f"正在使用 {strategy_class.display_name} 策略回测..."):
            try:
                # 创建策略实例
                strategy_instance = strategy_class(params)
                # 使用新的回测函数
                result = run_backtest_with_strategy(df_data, symbol, strategy_instance, initial_capital)
                st.session_state['result'] = result
                st.session_state['df_data'] = df_data
                st.session_state['params'] = params
                st.session_state['strategy_class'] = strategy_class
                st.success(f"✅ 回测完成! 共 {len(result.trades)} 笔交易")
            except Exception as e:
                st.error(f"回测失败: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # 显示结果
    if 'result' in st.session_state:
        result = st.session_state['result']
        df_data = st.session_state['df_data']
        params = st.session_state['params']
        strategy_class = st.session_state.get('strategy_class', None)

        # 标签页
        tabs = st.tabs(["📊 概览", "💹 资金曲线", "📈 K线分析", "📋 交易记录", "📉 统计分析"])

        with tabs[0]:
            render_overview(result)

        with tabs[1]:
            render_equity_chart(result)

        with tabs[2]:
            if strategy_class:
                render_kline_analysis(result, df_data, params, strategy_class)
            else:
                st.warning("需要策略类信息来渲染K线分析")

        with tabs[3]:
            render_trades_table(result)

        with tabs[4]:
            render_statistics(result)

    else:
        st.info("👈 请在侧边栏配置参数后点击「运行回测」")

        # 显示品种信息
        st.subheader("📌 支持的品种")
        inst_data = []
        for sym, inst in INSTRUMENTS.items():
            inst_data.append({
                '代码': sym,
                '名称': inst['name'],
                '交易所': inst['exchange'],
                '乘数': inst['multiplier'],
                '最小变动': inst['price_tick'],
                '保证金': f"{inst['margin_rate']*100:.0f}%",
                '夜盘': '是' if inst['night_trade'] else '否'
            })
        st.dataframe(pd.DataFrame(inst_data), use_container_width=True, hide_index=True)


if __name__ == '__main__':
    main()
