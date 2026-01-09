# coding=utf-8
"""
多因子选股回测页面
支持沪深300/中证500/中证1000成分股
使用LightGBM排序模型
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import threading
import queue

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def render_multifactor_page():
    """渲染多因子选股页面"""
    st.title("多因子选股回测")

    # 子页面选择
    sub_page = st.radio(
        "功能选择",
        ["策略回测", "因子分析", "模型训练"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    if sub_page == "策略回测":
        _render_backtest_page()
    elif sub_page == "因子分析":
        _render_factor_analysis()
    else:
        _render_model_training()


def _render_backtest_page():
    """渲染回测页面"""
    # 三列布局
    col1, col2, col3 = st.columns([1, 1.2, 0.8])

    with col1:
        st.subheader("回测配置")

        # 指数选择
        index_options = {
            "zz1000": "中证1000 (1000只)",
            "zz500": "中证500 (500只)",
            "hs300": "沪深300 (300只)"
        }
        index_name = st.selectbox(
            "股票池",
            options=list(index_options.keys()),
            format_func=lambda x: index_options[x],
            index=0
        )

        # 时间范围
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input(
                "开始日期",
                value=datetime(2021, 1, 1),
                min_value=datetime(2015, 1, 1),
                max_value=datetime.now()
            )
        with col_b:
            end_date = st.date_input(
                "结束日期",
                value=datetime(2025, 12, 31),
                min_value=datetime(2015, 1, 1),
                max_value=datetime.now()
            )

        # 持仓设置
        hold_num = st.slider("持仓数量", min_value=5, max_value=50, value=20)
        rebalance_num = st.slider("每日换仓数", min_value=1, max_value=10, value=2)

        # 模型设置
        train_days = st.number_input("训练窗口(天)", min_value=60, max_value=500, value=120)
        retrain_freq = st.number_input("重训频率(天)", min_value=5, max_value=60, value=20)

        # 初始资金
        initial_capital = st.number_input(
            "初始资金",
            min_value=100000,
            max_value=10000000,
            value=1000000,
            step=100000
        )

        # 最大股票数限制
        max_stocks = st.number_input(
            "股票数限制 (0=不限)",
            min_value=0,
            max_value=1000,
            value=0,
            help="限制股票池大小，用于快速测试"
        )

        st.markdown("---")

        # 运行按钮
        run_btn = st.button("开始回测", type="primary", use_container_width=True)

    with col2:
        st.subheader("因子配置")

        # 因子分组
        factor_groups = {
            "动量因子": ["mom_5d", "mom_10d", "mom_20d", "mom_60d", "reversal_5d"],
            "波动因子": ["vol_5d", "vol_20d", "vol_60d", "vol_ratio"],
            "成交量因子": ["vol_ma_ratio", "amount_ma_ratio"],
            "价格因子": ["price_to_high", "price_to_low", "price_position"],
            "均线因子": ["ma_bias_5", "ma_bias_20", "ma_bias_60", "ma_bull"],
            "技术指标": ["rsi", "macd", "macd_hist"]
        }

        selected_factors = []
        for group_name, factors in factor_groups.items():
            with st.expander(group_name, expanded=True):
                cols = st.columns(3)
                for i, factor in enumerate(factors):
                    with cols[i % 3]:
                        if st.checkbox(factor, value=True, key=f"factor_{factor}"):
                            selected_factors.append(factor)

        st.caption(f"已选择 {len(selected_factors)} 个因子")

    with col3:
        st.subheader("策略说明")

        st.markdown("""
        **多因子选股策略**

        使用LightGBM排序模型预测股票未来收益，选取预测得分最高的股票持有。

        **核心逻辑**
        1. 计算所有股票的因子值
        2. 使用历史数据训练LightGBM模型
        3. 预测各股票未来5日收益
        4. 选取预测得分Top N持有
        5. 每日换仓M只

        **风险提示**
        - 历史回测不代表未来
        - 小盘股流动性风险
        - 换手率成本影响
        """)

        st.markdown("---")
        st.caption("模型: LightGBM GBDT")
        st.caption("目标: 5日收益率")
        st.caption("训练: 滚动窗口")

    st.markdown("---")

    # 运行回测
    if run_btn:
        _run_multifactor_backtest(
            index_name=index_name,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            hold_num=hold_num,
            rebalance_num=rebalance_num,
            train_days=train_days,
            retrain_freq=retrain_freq,
            initial_capital=initial_capital,
            max_stocks=max_stocks,
            selected_factors=selected_factors
        )

    # 显示结果
    if 'multifactor_result' in st.session_state and st.session_state.multifactor_result:
        _render_backtest_results()


def _run_multifactor_backtest(index_name, start_date, end_date, hold_num,
                               rebalance_num, train_days, retrain_freq,
                               initial_capital, max_stocks, selected_factors):
    """运行多因子回测"""
    try:
        from multifactor.run_multifactor import run_multifactor_strategy
        from multifactor.data_loader import get_index_components

        # 获取股票池大小
        stock_pool = get_index_components(index_name)
        pool_size = len(stock_pool)
        if max_stocks > 0:
            pool_size = min(pool_size, max_stocks)

        st.info(f"准备回测: {index_name} ({pool_size}只), {start_date} ~ {end_date}")

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("正在加载数据和计算因子...")
        progress_bar.progress(10)

        # 运行回测
        with st.spinner("正在运行多因子回测，这可能需要几分钟..."):
            result = run_multifactor_strategy(
                start_date=start_date,
                end_date=end_date,
                hold_num=hold_num,
                rebalance_num=rebalance_num,
                train_days=train_days,
                retrain_freq=retrain_freq,
                use_sample=False,
                index_name=index_name,
                max_stocks=max_stocks
            )

        progress_bar.progress(100)
        status_text.empty()

        if result:
            st.session_state.multifactor_result = result
            st.success(f"回测完成! 累计收益: {result.total_return*100:.2f}%, 夏普: {result.sharpe_ratio:.2f}")
        else:
            st.error("回测失败，请检查数据")

    except Exception as e:
        st.error(f"回测出错: {e}")
        import traceback
        st.code(traceback.format_exc())


def _render_backtest_results():
    """渲染回测结果"""
    result = st.session_state.multifactor_result

    # 标签页
    tabs = st.tabs(["概览", "资金曲线", "交易记录", "因子分析"])

    with tabs[0]:
        _render_overview(result)

    with tabs[1]:
        _render_equity_curve(result)

    with tabs[2]:
        _render_trades(result)

    with tabs[3]:
        _render_factor_importance(result)


def _render_overview(result):
    """渲染概览"""
    st.subheader("回测概览")

    # 关键指标
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("累计收益", f"{result.total_return*100:.2f}%")
    with col2:
        st.metric("年化收益", f"{result.annual_return*100:.2f}%")
    with col3:
        st.metric("最大回撤", f"{result.max_drawdown*100:.2f}%")
    with col4:
        st.metric("夏普比率", f"{result.sharpe_ratio:.2f}")
    with col5:
        st.metric("胜率", f"{result.win_rate*100:.1f}%")
    with col6:
        st.metric("年换手率", f"{result.turnover:.1f}x")

    st.markdown("---")

    # 详细指标
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**收益指标**")
        st.write(f"- 初始资金: ¥{result.initial_capital:,.0f}")
        st.write(f"- 期末资金: ¥{result.final_value:,.0f}")
        st.write(f"- 总收益率: {result.total_return*100:.2f}%")
        st.write(f"- 年化收益: {result.annual_return*100:.2f}%")

    with col2:
        st.write("**风险指标**")
        st.write(f"- 最大回撤: {result.max_drawdown*100:.2f}%")
        st.write(f"- 夏普比率: {result.sharpe_ratio:.2f}")
        st.write(f"- 卡玛比率: {result.calmar_ratio:.2f}")

    with col3:
        st.write("**交易统计**")
        st.write(f"- 总交易次数: {result.total_trades}")
        st.write(f"- 胜率: {result.win_rate*100:.1f}%")
        st.write(f"- 年换手率: {result.turnover:.1f}倍")


def _render_equity_curve(result):
    """渲染资金曲线"""
    st.subheader("资金曲线")

    if result.equity_curve is None or result.equity_curve.empty:
        st.warning("没有资金曲线数据")
        return

    df = result.equity_curve

    # 创建图表
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('账户净值', '回撤')
    )

    # 净值曲线
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['total_value'],
            name='账户净值',
            line=dict(color='#2196F3', width=2),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.1)'
        ),
        row=1, col=1
    )

    # 回撤曲线
    rolling_max = df['total_value'].cummax()
    drawdown = (df['total_value'] - rolling_max) / rolling_max * 100

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=drawdown,
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


def _render_trades(result):
    """渲染交易记录"""
    st.subheader("交易记录")

    if not result.trades:
        st.warning("没有交易记录")
        return

    # 转换为DataFrame
    trades_data = []
    for t in result.trades:
        trades_data.append({
            '日期': t.date,
            '代码': t.code,
            '方向': t.direction,
            '价格': f"{t.price:.2f}",
            '数量': t.shares,
            '金额': f"¥{t.amount:,.0f}",
            '手续费': f"¥{t.commission:.2f}",
            '盈亏': f"¥{t.pnl:,.0f}" if hasattr(t, 'pnl') else "-"
        })

    df_trades = pd.DataFrame(trades_data)

    # 筛选
    col1, col2 = st.columns(2)
    with col1:
        direction_filter = st.multiselect(
            "方向筛选",
            options=df_trades['方向'].unique().tolist(),
            default=df_trades['方向'].unique().tolist()
        )

    df_filtered = df_trades[df_trades['方向'].isin(direction_filter)]

    st.dataframe(df_filtered, use_container_width=True, hide_index=True)

    # 统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("买入次数", len(df_trades[df_trades['方向'] == 'BUY']))
    with col2:
        st.metric("卖出次数", len(df_trades[df_trades['方向'] == 'SELL']))
    with col3:
        total_commission = sum(t.commission for t in result.trades)
        st.metric("总手续费", f"¥{total_commission:,.0f}")


def _render_factor_importance(result):
    """渲染因子重要性分析"""
    st.subheader("因子分析")

    st.info("因子重要性分析需要完整的回测结果数据")

    # 显示使用的因子
    from multifactor.factors import get_factor_list
    factors = get_factor_list()

    st.write("**使用的因子列表**")

    factor_groups = {
        "动量因子": ["mom_5d", "mom_10d", "mom_20d", "mom_60d", "reversal_5d"],
        "波动因子": ["vol_5d", "vol_20d", "vol_60d", "vol_ratio"],
        "成交量因子": ["vol_ma_ratio", "amount_ma_ratio"],
        "价格因子": ["price_to_high", "price_to_low", "price_position"],
        "均线因子": ["ma_bias_5", "ma_bias_20", "ma_bias_60", "ma_bull"],
        "技术指标": ["rsi", "macd", "macd_hist"]
    }

    for group_name, group_factors in factor_groups.items():
        used = [f for f in group_factors if f in factors]
        st.write(f"**{group_name}**: {', '.join(used)}")


def _render_factor_analysis():
    """因子分析页面"""
    st.subheader("因子分析")

    st.info("此功能正在开发中，敬请期待...")

    st.markdown("""
    **计划功能**
    - 单因子IC分析
    - 因子相关性矩阵
    - 因子分层回测
    - 因子衰减分析
    """)


def _render_model_training():
    """模型训练页面"""
    st.subheader("模型训练")

    st.info("此功能正在开发中，敬请期待...")

    st.markdown("""
    **计划功能**
    - 自定义模型参数
    - 交叉验证
    - 超参数调优
    - 模型保存/加载
    """)


if __name__ == "__main__":
    st.set_page_config(page_title="多因子选股", layout="wide")
    render_multifactor_page()
