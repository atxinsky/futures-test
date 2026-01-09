# coding=utf-8
"""
ETF回测页面
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def render_etf_backtest_page():
    """渲染ETF回测页面"""
    st.markdown("## 📈 ETF策略回测")

    from core.etf_data_service import ETF_POOLS, ALL_ETFS, BIGBROTHER_POOL

    # 三列布局
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### 📅 回测设置")

        start_date = st.date_input(
            "开始日期",
            value=datetime(2021, 1, 1),
            min_value=datetime(2019, 1, 1),
            max_value=datetime.now()
        )

        end_date = st.date_input(
            "结束日期",
            value=datetime.now(),
            min_value=datetime(2019, 1, 1),
            max_value=datetime.now()
        )

        initial_capital = st.number_input(
            "初始资金",
            min_value=100000,
            max_value=100000000,
            value=1000000,
            step=100000
        )

        commission = st.number_input(
            "手续费率",
            min_value=0.0,
            max_value=0.01,
            value=0.0001,
            step=0.0001,
            format="%.4f"
        )

    with col2:
        st.markdown("### ⚙️ 策略参数")

        strategy_options = [
            "BigBrother V14 (EMA金叉+ADX)",
            "BigBrother V17 (Donchian经典)",
            "BigBrother V19 (Donchian科技)",
            "BigBrother V20 (Donchian均衡)",
            "BigBrother V21 (Donchian防跳空)"
        ]
        strategy_name = st.selectbox("策略", strategy_options)

        # 根据策略类型显示不同参数
        if "V14" in strategy_name:
            base_position = st.slider("基础仓位", 0.05, 0.30, 0.18, 0.01)
            max_loss = st.slider("硬止损比例", 0.05, 0.15, 0.07, 0.01)
            atr_multiplier = st.slider("ATR止损倍数", 1.5, 4.0, 2.5, 0.1)
            trail_start = st.slider("追踪止盈触发", 0.08, 0.30, 0.15, 0.01)
            adx_threshold = st.slider("ADX阈值", 15, 30, 20, 1)
            strategy_params = {
                "base_position": base_position,
                "max_loss": max_loss,
                "atr_multiplier": atr_multiplier,
                "trail_start": trail_start,
                "adx_threshold": adx_threshold
            }
        else:
            # V17-V21 使用 Donchian Channel 参数
            if "V17" in strategy_name:
                risk_default, max_pos_default = 0.01, 0.25
            elif "V19" in strategy_name:
                risk_default, max_pos_default = 0.012, 0.22
            else:  # V20, V21
                risk_default, max_pos_default = 0.01, 0.30

            risk_per_trade = st.slider("单笔风险", 0.005, 0.03, risk_default, 0.002)
            max_position = st.slider("最大仓位", 0.10, 0.40, max_pos_default, 0.05)
            donchian_high = st.slider("突破周期", 10, 30, 20, 5)
            donchian_low = st.slider("跌破周期", 5, 20, 10, 5)

            strategy_params = {
                "risk_per_trade": risk_per_trade,
                "max_position": max_position,
                "donchian_high_period": donchian_high,
                "donchian_low_period": donchian_low
            }

            if "V21" in strategy_name:
                gap_up = st.slider("高开限制", 0.01, 0.05, 0.02, 0.005)
                strategy_params["gap_up_limit"] = gap_up

    with col3:
        st.markdown("### 📋 标的池")

        pool_options = ["BigBrother V14 默认池"] + list(ETF_POOLS.keys()) + ["自定义"]
        selected_pool = st.selectbox("预设池", pool_options)

        if selected_pool == "BigBrother V14 默认池":
            default_codes = BIGBROTHER_POOL
        elif selected_pool == "自定义":
            default_codes = []
        else:
            default_codes = list(ETF_POOLS[selected_pool].keys())

        selected_etfs = st.multiselect(
            "选择ETF",
            options=list(ALL_ETFS.keys()),
            default=default_codes,
            format_func=lambda x: f"{x} - {ALL_ETFS.get(x, '')}"
        )

        benchmark = st.selectbox(
            "基准",
            ["510300.SH (沪深300ETF)", "000300.SH (沪深300指数)"]
        )

    st.markdown("---")

    if st.button("🚀 运行回测", type="primary", use_container_width=True):
        if not selected_etfs:
            st.error("请至少选择一个ETF")
            return

        _run_etf_backtest(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_capital=initial_capital,
            commission=commission,
            selected_etfs=selected_etfs,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            benchmark=benchmark.split(" ")[0]
        )


def _run_etf_backtest(start_date, end_date, initial_capital, commission,
                      selected_etfs, strategy_name, strategy_params, benchmark):
    """运行ETF回测"""

    with st.spinner("正在加载数据..."):
        try:
            from core.etf_data_service import get_etf_data_service, ALL_ETFS
            from core.etf_backtest_engine import ETFBacktestEngine
            from strategies.etf_bigbrother_v14 import ETFBigBrotherV14
            from strategies.etf_bigbrother_v17_v21 import (
                ETFBigBrotherV17, ETFBigBrotherV19, ETFBigBrotherV20, ETFBigBrotherV21
            )

            ds = get_etf_data_service()

            all_codes = selected_etfs + [benchmark, "000300.SH"]
            all_codes = list(set(all_codes))

            data = {}
            progress_bar = st.progress(0)

            for i, code in enumerate(all_codes):
                df = ds.get_data_with_indicators(code, start_date, end_date)

                if len(df) == 0:
                    st.warning(f"无数据: {code}，正在从网络获取...")
                    ds.update_data(code)
                    df = ds.get_data_with_indicators(code, start_date, end_date)

                if len(df) > 0:
                    data[code] = df

                progress_bar.progress((i + 1) / len(all_codes))

            progress_bar.empty()

            if not data:
                st.error("无法加载数据")
                return

            st.success(f"数据加载完成: {len(data)}个标的")

        except ImportError as e:
            st.error(f"模块导入失败: {e}")
            st.info("请确保已安装: pip install akshare")
            return
        except Exception as e:
            st.error(f"数据加载失败: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    with st.spinner("正在运行回测..."):
        try:
            # 根据策略名称创建策略实例
            if "V14" in strategy_name:
                strategy = ETFBigBrotherV14(pool=selected_etfs, **strategy_params)
            elif "V17" in strategy_name:
                strategy = ETFBigBrotherV17(pool=selected_etfs, **strategy_params)
            elif "V19" in strategy_name:
                strategy = ETFBigBrotherV19(pool=selected_etfs, **strategy_params)
            elif "V20" in strategy_name:
                strategy = ETFBigBrotherV20(pool=selected_etfs, **strategy_params)
            elif "V21" in strategy_name:
                strategy = ETFBigBrotherV21(pool=selected_etfs, **strategy_params)
            else:
                strategy = ETFBigBrotherV14(pool=selected_etfs, **strategy_params)

            engine = ETFBacktestEngine(
                initial_capital=initial_capital,
                commission_rate=commission,
                slippage=0.0001,
                benchmark=benchmark
            )

            engine.set_strategy(strategy.initialize, strategy.handle_data)

            result = engine.run(
                data=data,
                start_date=start_date,
                end_date=end_date,
                benchmark_data=data.get(benchmark, data.get("000300.SH"))
            )

            st.success("回测完成!")
            _display_etf_result(result)

        except Exception as e:
            st.error(f"回测失败: {e}")
            import traceback
            st.code(traceback.format_exc())


def _display_etf_result(result):
    """显示ETF回测结果"""

    st.markdown("### 📊 绩效概览")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("累计收益", f"{result.total_return*100:.2f}%",
                  delta=f"vs基准 {result.excess_return*100:+.2f}%")
    with col2:
        st.metric("年化收益", f"{result.annual_return*100:.2f}%")
    with col3:
        st.metric("最大回撤", f"{result.max_drawdown*100:.2f}%")
    with col4:
        st.metric("夏普比率", f"{result.sharpe_ratio:.2f}")
    with col5:
        st.metric("胜率", f"{result.win_rate*100:.1f}%")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("基准收益", f"{result.benchmark_return*100:.2f}%")
    with col2:
        st.metric("波动率", f"{result.volatility*100:.2f}%")
    with col3:
        st.metric("卡玛比率", f"{result.calmar_ratio:.2f}")
    with col4:
        st.metric("盈亏比", f"{result.profit_loss_ratio:.2f}")
    with col5:
        st.metric("总交易次数", f"{result.total_trades}")

    st.markdown("---")

    # 权益曲线
    st.markdown("### 📈 权益曲线")

    if result.equity_curve is not None:
        df = result.equity_curve.reset_index()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.7, 0.3], subplot_titles=("累计收益率", "每日收益率"))

        fig.add_trace(go.Scatter(x=df["date"], y=df["cumulative_return"] * 100,
                                  mode="lines", name="策略收益", line=dict(color="#1f77b4", width=2)),
                      row=1, col=1)

        colors = ["#00c853" if r >= 0 else "#ff1744" for r in df["return"].fillna(0)]
        fig.add_trace(go.Bar(x=df["date"], y=df["return"] * 100, name="每日收益", marker_color=colors),
                      row=2, col=1)

        fig.update_layout(height=500, showlegend=True, hovermode="x unified")
        fig.update_yaxes(title_text="收益率 (%)", row=1, col=1)
        fig.update_yaxes(title_text="日收益 (%)", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    # 回撤曲线
    st.markdown("### 📉 回撤曲线")

    if result.equity_curve is not None:
        df = result.equity_curve.reset_index()
        rolling_max = df["total_value"].cummax()
        drawdown = (df["total_value"] - rolling_max) / rolling_max * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=df["date"], y=drawdown, mode="lines", fill="tozeroy",
                                     name="回撤", line=dict(color="#ff1744", width=1),
                                     fillcolor="rgba(255, 23, 68, 0.3)"))
        fig_dd.update_layout(height=250, showlegend=False, yaxis_title="回撤 (%)")

        st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("---")

    # 详细统计
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 收益统计")
        stats_df = pd.DataFrame({
            "指标": ["累计收益率", "年化收益率", "基准收益率", "超额收益",
                     "波动率", "最大回撤", "最大回撤持续天数"],
            "数值": [f"{result.total_return*100:.2f}%", f"{result.annual_return*100:.2f}%",
                     f"{result.benchmark_return*100:.2f}%", f"{result.excess_return*100:.2f}%",
                     f"{result.volatility*100:.2f}%", f"{result.max_drawdown*100:.2f}%",
                     f"{result.max_drawdown_duration}天"]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 📊 交易统计")
        trade_df = pd.DataFrame({
            "指标": ["总交易次数", "盈利次数", "亏损次数", "胜率",
                     "盈亏比", "平均盈利", "平均亏损", "平均持仓天数"],
            "数值": [f"{result.total_trades}", f"{result.win_trades}", f"{result.lose_trades}",
                     f"{result.win_rate*100:.1f}%", f"{result.profit_loss_ratio:.2f}",
                     f"¥{result.avg_win:,.0f}", f"¥{result.avg_loss:,.0f}",
                     f"{result.avg_holding_days:.1f}天"]
        })
        st.dataframe(trade_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 交易记录
    st.markdown("### 📋 交易记录")

    if result.trades:
        from core.etf_data_service import ALL_ETFS

        trades_data = []
        for t in result.trades:
            trades_data.append({
                "日期": t.date,
                "代码": t.code,
                "名称": ALL_ETFS.get(t.code, ""),
                "方向": "买入" if t.direction == "BUY" else "卖出",
                "价格": f"{t.price:.3f}",
                "股数": t.shares,
                "金额": f"¥{t.amount:,.0f}",
                "盈亏": f"¥{t.pnl:,.0f}" if t.direction == "SELL" else "-",
                "盈亏%": f"{t.pnl_pct*100:+.2f}%" if t.direction == "SELL" else "-",
            })

        trades_df = pd.DataFrame(trades_data)

        col1, col2 = st.columns(2)
        with col1:
            direction_filter = st.selectbox("方向筛选", ["全部", "买入", "卖出"])
        with col2:
            code_filter = st.selectbox("标的筛选", ["全部"] + list(set([t.code for t in result.trades])))

        if direction_filter != "全部":
            trades_df = trades_df[trades_df["方向"] == direction_filter]
        if code_filter != "全部":
            trades_df = trades_df[trades_df["代码"] == code_filter]

        st.dataframe(trades_df, use_container_width=True, hide_index=True)

        csv = trades_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载交易记录", csv, "etf_trades.csv", "text/csv")
