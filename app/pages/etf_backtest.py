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
    st.markdown("## ETF策略回测")

    from core.etf_data_service import ETF_POOLS, ALL_ETFS, BIGBROTHER_POOL

    # 检测是否有优化参数待应用
    if 'opt_apply_params' in st.session_state and st.session_state['opt_apply_params']:
        opt = st.session_state['opt_apply_params']
        with st.container():
            st.info(f"""
            **检测到优化参数可应用**  
            策略: {opt['strategy']} | 目标: {opt['opt_target']}={opt['best_value']:.3f}  
            标的池: {len(opt['etf_pool'])}个ETF | 训练集: {opt['train_range']}
            """)
            col_apply1, col_apply2, col_apply3 = st.columns([1, 1, 2])
            with col_apply1:
                if st.button("✅ 应用参数", type="primary", key="apply_opt_params"):
                    # 保存到 applied_params 供 slider 使用
                    st.session_state['applied_opt_params'] = opt.copy()
                    st.session_state['opt_apply_params'] = None  # 清除待应用状态
                    st.rerun()
            with col_apply2:
                if st.button("❌ 忽略", key="ignore_opt_params"):
                    st.session_state['opt_apply_params'] = None
                    st.rerun()
        st.markdown("---")

    # 获取已应用的优化参数（用于设置默认值）
    applied = st.session_state.get('applied_opt_params', {})
    applied_params = applied.get('params', {})
    applied_pool = applied.get('etf_pool', [])
    applied_strategy = applied.get('strategy', '')

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

        # 根据优化参数自动选择策略
        strategy_options = [
            "BigBrother V14 (EMA金叉+ADX)",
            "BigBrother V17 (Donchian经典)",
            "BigBrother V19 (Donchian科技)",
            "BigBrother V20 (Donchian均衡)",
            "BigBrother V21 (Donchian防跳空)"
        ]

        # 确定默认策略索引
        default_strategy_idx = 0
        if applied_strategy:
            for i, opt in enumerate(strategy_options):
                if "V14" in applied_strategy and "V14" in opt:
                    default_strategy_idx = i
                    break
                elif "V17" in applied_strategy and "V17" in opt:
                    default_strategy_idx = i
                    break
                elif "V21" in applied_strategy and "V21" in opt:
                    default_strategy_idx = i
                    break

        strategy_name = st.selectbox("策略", strategy_options, index=default_strategy_idx)

        # 显示已应用优化参数提示
        if applied_params:
            st.success(f"已应用优化参数 (可调整)")

        # 根据策略类型显示不同参数
        if "V14" in strategy_name:
            base_position = st.slider("基础仓位", 0.05, 0.30, 
                                      applied_params.get('base_position', 0.18), 0.01,
                                      key="v14_base_pos")
            max_loss = st.slider("硬止损比例", 0.05, 0.15, 
                                 applied_params.get('max_loss', 0.07), 0.01,
                                 key="v14_max_loss")
            atr_multiplier = st.slider("ATR止损倍数", 1.5, 4.0, 
                                       applied_params.get('atr_multiplier', 2.5), 0.1,
                                       key="v14_atr_mult")
            trail_start = st.slider("追踪止盈触发", 0.08, 0.30, 
                                    applied_params.get('trail_start', 0.15), 0.01,
                                    key="v14_trail_start")
            adx_threshold = st.slider("ADX阈值", 15, 30, 
                                      int(applied_params.get('adx_threshold', 20)), 1,
                                      key="v14_adx")
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

            # 如果有优化参数，使用优化后的值
            risk_val = applied_params.get('risk_per_trade', risk_default)
            max_pos_val = applied_params.get('max_position', max_pos_default)
            dc_high_val = int(applied_params.get('donchian_high_period', 20))
            dc_low_val = int(applied_params.get('donchian_low_period', 10))

            risk_per_trade = st.slider("单笔风险", 0.005, 0.03, risk_val, 0.002, key="dc_risk")
            max_position = st.slider("最大仓位", 0.10, 0.40, max_pos_val, 0.05, key="dc_max_pos")
            donchian_high = st.slider("突破周期", 10, 40, dc_high_val, 1, key="dc_high")
            donchian_low = st.slider("跌破周期", 5, 25, dc_low_val, 1, key="dc_low")

            strategy_params = {
                "risk_per_trade": risk_per_trade,
                "max_position": max_position,
                "donchian_high_period": donchian_high,
                "donchian_low_period": donchian_low
            }

            if "V21" in strategy_name:
                gap_val = applied_params.get('gap_up_limit', 0.02)
                gap_up = st.slider("高开限制", 0.01, 0.05, gap_val, 0.005, key="dc_gap")
                strategy_params["gap_up_limit"] = gap_up

    with col3:
        st.markdown("### 📋 标的池")

        # 如果有优化参数应用，添加"优化参数池"选项
        pool_options = ["BigBrother V14 默认池"] + list(ETF_POOLS.keys()) + ["自定义"]
        if applied_pool:
            pool_options = ["优化参数池"] + pool_options

        # 默认选择优化参数池（如果有）
        default_pool_idx = 0

        selected_pool = st.selectbox("预设池", pool_options, index=default_pool_idx, key="etf_pool_select")

        if selected_pool == "优化参数池" and applied_pool:
            default_codes = applied_pool
            st.caption(f"来自优化结果: {len(applied_pool)}个ETF")
        elif selected_pool == "BigBrother V14 默认池":
            default_codes = BIGBROTHER_POOL
        elif selected_pool == "自定义":
            default_codes = []
        else:
            default_codes = list(ETF_POOLS[selected_pool].keys())

        selected_etfs = st.multiselect(
            "选择ETF",
            options=list(ALL_ETFS.keys()),
            default=default_codes,
            format_func=lambda x: f"{x} - {ALL_ETFS.get(x, '')}",
            key="etf_multiselect"
        )

        benchmark = st.selectbox(
            "基准",
            ["510300.SH (沪深300ETF)", "000300.SH (沪深300指数)"]
        )

    st.markdown("---")

    # 检查是否有已保存的回测结果
    has_result = 'etf_backtest_result' in st.session_state and st.session_state['etf_backtest_result'] is not None

    col_btn, col_status = st.columns([3, 1])
    with col_btn:
        run_clicked = st.button("🚀 运行回测", type="primary", use_container_width=True)
    with col_status:
        if has_result:
            st.success("已有回测结果")

    if run_clicked:
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
    # 页面rerun时，如果session_state中有已保存的回测结果，继续显示
    elif has_result:
        result = st.session_state['etf_backtest_result']
        data = st.session_state.get('etf_backtest_data')
        _display_etf_result(result, data)


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

            # 只加载选中的ETF和基准，不强制加载000300.SH指数
            all_codes = selected_etfs + [benchmark]
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
                benchmark_data=data.get(benchmark, data.get("510300.SH"))  # 使用沪深300ETF作为fallback
            )

            st.success("回测完成!")
            # 保存数据到session_state供K线图使用和保存功能（使用etf_前缀避免与期货回测冲突）
            st.session_state['etf_backtest_data'] = data
            st.session_state['etf_backtest_result'] = result
            st.session_state['etf_backtest_config'] = {
                'strategy_name': strategy_name,
                'selected_etfs': selected_etfs,
                'strategy_params': strategy_params
            }
            _display_etf_result(result, data)

        except Exception as e:
            st.error(f"回测失败: {e}")
            import traceback
            st.code(traceback.format_exc())


def _display_etf_result(result, data=None):
    """显示ETF回测结果"""

    # 保存按钮
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        notes = st.text_input("备注", key="save_notes", placeholder="可选：添加备注")
    with col3:
        if st.button("💾 保存回测结果", type="primary"):
            _save_backtest_result(notes)

    # 使用radio代替tabs，这样可以通过key保持选择状态
    tab_options = ["概览", "K线交易图", "资金曲线", "交易记录", "统计分析"]
    selected_tab = st.radio(
        "结果视图",
        tab_options,
        horizontal=True,
        key="etf_result_tab",
        label_visibility="collapsed"
    )

    st.markdown("---")

    if selected_tab == "概览":
        _render_overview_tab(result)
    elif selected_tab == "K线交易图":
        _render_kline_trade_chart(result, st.session_state.get('etf_backtest_data'))
    elif selected_tab == "资金曲线":
        _render_equity_curve_tab(result)
    elif selected_tab == "交易记录":
        _render_trades_tab(result)
    elif selected_tab == "统计分析":
        _render_statistics_tab(result)


def _save_backtest_result(notes: str = ""):
    """保存回测结果到数据库"""
    result = st.session_state.get('etf_backtest_result')
    config = st.session_state.get('etf_backtest_config')

    if not result or not config:
        st.error("没有可保存的回测结果")
        return

    try:
        from utils.backtest_storage import get_backtest_storage

        storage = get_backtest_storage()
        backtest_id = storage.save_etf_backtest(
            result=result,
            strategy_name=config['strategy_name'],
            symbols=config['selected_etfs'],
            params=config['strategy_params'],
            notes=notes
        )

        st.success(f"回测已保存! ID: {backtest_id}")
        st.info("可在「回测历史」页面查看所有保存的回测记录")

    except Exception as e:
        st.error(f"保存失败: {e}")


def _render_overview_tab(result):
    """概览标签页"""
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


def _render_equity_curve_tab(result):
    """资金曲线标签页"""
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


def _render_statistics_tab(result):
    """统计分析标签页"""
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

    # 月度收益分析
    st.markdown("---")
    st.markdown("### 📅 月度收益分析")

    if result.equity_curve is not None:
        df = result.equity_curve.reset_index()
        if 'date' in df.columns:
            df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
            monthly = df.groupby('month').agg({
                'total_value': ['first', 'last']
            })
            monthly.columns = ['start_value', 'end_value']
            monthly['return'] = (monthly['end_value'] - monthly['start_value']) / monthly['start_value'] * 100
            monthly = monthly.reset_index()
            monthly['month'] = monthly['month'].astype(str)

            # 月度收益柱状图
            colors = ['#4CAF50' if r >= 0 else '#F44336' for r in monthly['return']]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly['month'],
                y=monthly['return'],
                marker_color=colors,
                text=[f"{r:+.1f}%" for r in monthly['return']],
                textposition='outside'
            ))
            fig.update_layout(
                height=300,
                yaxis_title="月收益率 (%)",
                showlegend=False,
                margin=dict(l=50, r=50, t=30, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)

            # 月度收益表格
            monthly_display = monthly[['month', 'return']].copy()
            monthly_display.columns = ['月份', '收益率']
            monthly_display['收益率'] = monthly_display['收益率'].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(monthly_display, use_container_width=True, hide_index=True)


def _render_kline_trade_chart(result, data):
    """K线交易图标签页"""
    st.markdown("### 📊 K线交易图")

    if not result.trades:
        st.warning("没有交易记录，无法显示K线图")
        return

    if not data:
        st.warning("没有K线数据")
        return

    from core.etf_data_service import ALL_ETFS

    # 获取交易过的ETF列表
    traded_codes = list(set([t.code for t in result.trades]))

    if not traded_codes:
        st.warning("没有交易记录")
        return

    # 选择要显示的ETF
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        selected_code = st.selectbox(
            "选择标的",
            options=traded_codes,
            format_func=lambda x: f"{x} - {ALL_ETFS.get(x, '')}",
            key="kline_etf_select"
        )

    # 获取该ETF的数据
    if selected_code not in data:
        st.warning(f"没有 {selected_code} 的K线数据")
        return

    df_data = data[selected_code].copy()
    if 'date' not in df_data.columns:
        df_data = df_data.reset_index()

    # 确保date列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df_data['date']):
        df_data['date'] = pd.to_datetime(df_data['date'])

    # 筛选该ETF的交易，配对买卖
    code_trades = [t for t in result.trades if t.code == selected_code]
    trade_pairs = _match_trade_pairs(code_trades)

    with col2:
        result_filter = st.multiselect(
            "筛选结果",
            options=['盈利', '亏损'],
            default=['盈利', '亏损'],
            key="etf_kline_result_filter"
        )

    with col3:
        if trade_pairs:
            trade_options = [
                f"#{i+1} {p['entry_date']}→{p['exit_date']} {'盈' if p['pnl'] > 0 else '亏'}{abs(p['pnl_pct']*100):.1f}%"
                for i, p in enumerate(trade_pairs)
            ]
            selected_trade_idx = st.selectbox(
                "跳转到交易",
                options=range(len(trade_options)),
                format_func=lambda x: trade_options[x],
                key="etf_kline_trade_select"
            )
        else:
            selected_trade_idx = 0

    # 筛选交易对
    filtered_pairs = [
        p for p in trade_pairs
        if (('盈利' in result_filter and p['pnl'] > 0) or ('亏损' in result_filter and p['pnl'] <= 0))
    ]

    st.write(f"显示 **{len(filtered_pairs)}** / {len(trade_pairs)} 笔交易")

    if len(df_data) == 0:
        st.warning("数据为空")
        return

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
            x=df_data['date'],
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

    # 添加EMA线（如果有）
    if 'ema_fast' in df_data.columns:
        fig.add_trace(
            go.Scatter(x=df_data['date'], y=df_data['ema_fast'], mode='lines',
                       name='EMA快', line=dict(color='#FF9800', width=1)),
            row=1, col=1
        )
    if 'ema_slow' in df_data.columns:
        fig.add_trace(
            go.Scatter(x=df_data['date'], y=df_data['ema_slow'], mode='lines',
                       name='EMA慢', line=dict(color='#2196F3', width=1)),
            row=1, col=1
        )

    # 成交量
    if 'volume' in df_data.columns:
        colors = ['#EF5350' if close >= open_p else '#26A69A'
                  for close, open_p in zip(df_data['close'], df_data['open'])]
        fig.add_trace(
            go.Bar(x=df_data['date'], y=df_data['volume'], name='成交量', marker_color=colors, opacity=0.7),
            row=2, col=1
        )

    # 持仓盈亏曲线
    holding_pnl = []
    holding_time = []
    for p in filtered_pairs:
        entry_dt = pd.to_datetime(p['entry_date'])
        exit_dt = pd.to_datetime(p['exit_date'])
        mask = (df_data['date'] >= entry_dt) & (df_data['date'] <= exit_dt)
        trade_data = df_data[mask]
        for _, row in trade_data.iterrows():
            pnl_pct = (row['close'] - p['entry_price']) / p['entry_price'] * 100
            holding_pnl.append(pnl_pct)
            holding_time.append(row['date'])

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
    for i, p in enumerate(filtered_pairs):
        is_win = p['pnl'] > 0
        entry_dt = pd.to_datetime(p['entry_date'])
        exit_dt = pd.to_datetime(p['exit_date'])

        # 入场标记
        entry_data = df_data[df_data['date'] == entry_dt]
        entry_y = entry_data['low'].values[0] * 0.995 if len(entry_data) > 0 else p['entry_price']

        fig.add_trace(
            go.Scatter(
                x=[entry_dt], y=[entry_y],
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=14, color='#2196F3', line=dict(color='white', width=1)),
                text=[f'买{p["shares"]}股'],
                textposition='bottom center',
                textfont=dict(size=9, color='#2196F3'),
                name=f'入场#{i+1}',
                showlegend=False,
                hovertemplate=f"<b>入场 #{i+1}</b><br>日期: {p['entry_date']}<br>价格: {p['entry_price']:.3f}<br>股数: {p['shares']}<extra></extra>"
            ),
            row=1, col=1
        )

        # 出场标记
        exit_data = df_data[df_data['date'] == exit_dt]
        exit_y = exit_data['high'].values[0] * 1.005 if len(exit_data) > 0 else p['exit_price']
        exit_color = '#4CAF50' if is_win else '#F44336'

        fig.add_trace(
            go.Scatter(
                x=[exit_dt], y=[exit_y],
                mode='markers+text',
                marker=dict(symbol='triangle-down', size=14, color=exit_color, line=dict(color='white', width=1)),
                text=[f'{p["pnl_pct"]*100:+.1f}%'],
                textposition='top center',
                textfont=dict(size=9, color=exit_color, weight='bold'),
                name=f'出场#{i+1}',
                showlegend=False,
                hovertemplate=f"<b>出场 #{i+1}</b><br>日期: {p['exit_date']}<br>价格: {p['exit_price']:.3f}<br>盈亏: ¥{p['pnl']:+,.0f} ({p['pnl_pct']*100:+.2f}%)<extra></extra>"
            ),
            row=1, col=1
        )

        # 连接线
        fig.add_trace(
            go.Scatter(
                x=[entry_dt, exit_dt],
                y=[p['entry_price'], p['exit_price']],
                mode='lines',
                line=dict(color=exit_color, width=2, dash='dot'),
                opacity=0.6, showlegend=False, hoverinfo='skip'
            ),
            row=1, col=1
        )

        # 持仓区间背景色
        fig.add_shape(
            type="rect",
            x0=entry_dt, x1=exit_dt,
            y0=y_min, y1=y_max,
            fillcolor='rgba(76, 175, 80, 0.15)' if is_win else 'rgba(244, 67, 54, 0.15)',
            layer='below', line_width=0,
            row=1, col=1
        )

    # 聚焦到选中的交易
    if selected_trade_idx is not None and selected_trade_idx < len(trade_pairs):
        selected_pair = trade_pairs[selected_trade_idx]
        trade_start = pd.to_datetime(selected_pair['entry_date'])
        trade_end = pd.to_datetime(selected_pair['exit_date'])

        try:
            start_idx = df_data[df_data['date'] <= trade_start].index[-1] - 20
            end_idx = df_data[df_data['date'] >= trade_end].index[0] + 20
            start_idx = max(0, start_idx)
            end_idx = min(len(df_data) - 1, end_idx)

            x_start = df_data.iloc[start_idx]['date']
            x_end = df_data.iloc[end_idx]['date']

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
        height=650,
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
    if selected_trade_idx is not None and selected_trade_idx < len(trade_pairs):
        p = trade_pairs[selected_trade_idx]
        st.markdown("---")
        st.write(f"### 交易 #{selected_trade_idx+1} 详情")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("入场日期", p['entry_date'])
            st.metric("入场价格", f"{p['entry_price']:.3f}")
        with col2:
            st.metric("出场日期", p['exit_date'])
            st.metric("出场价格", f"{p['exit_price']:.3f}")
        with col3:
            holding_days = (pd.to_datetime(p['exit_date']) - pd.to_datetime(p['entry_date'])).days
            st.metric("持仓天数", f"{holding_days}天")
            st.metric("交易股数", f"{p['shares']}股")
        with col4:
            st.metric("盈亏金额", f"¥{p['pnl']:+,.0f}", delta=f"{p['pnl_pct']*100:+.2f}%")


def _match_trade_pairs(trades):
    """
    将买卖交易配对成完整的交易对

    Args:
        trades: ETFTrade列表 (已按时间排序)

    Returns:
        交易对列表 [{entry_date, exit_date, entry_price, exit_price, shares, pnl, pnl_pct}, ...]
    """
    pairs = []
    open_position = None

    for t in trades:
        if t.direction == "BUY":
            # 开仓
            open_position = {
                'entry_date': t.date,
                'entry_price': t.price,
                'shares': t.shares
            }
        elif t.direction == "SELL" and open_position is not None:
            # 平仓
            pairs.append({
                'entry_date': open_position['entry_date'],
                'exit_date': t.date,
                'entry_price': open_position['entry_price'],
                'exit_price': t.price,
                'shares': open_position['shares'],
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct
            })
            open_position = None

    return pairs


def _render_trades_tab(result):
    """交易记录标签页"""
    st.markdown("### 📋 交易记录")

    if not result.trades:
        st.info("暂无交易记录")
        return

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
        direction_filter = st.selectbox("方向筛选", ["全部", "买入", "卖出"], key="trades_tab_dir_filter")
    with col2:
        code_filter = st.selectbox("标的筛选", ["全部"] + list(set([t.code for t in result.trades])), key="trades_tab_code_filter")

    if direction_filter != "全部":
        trades_df = trades_df[trades_df["方向"] == direction_filter]
    if code_filter != "全部":
        trades_df = trades_df[trades_df["代码"] == code_filter]

    st.dataframe(trades_df, use_container_width=True, hide_index=True)

    csv = trades_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 下载交易记录", csv, "etf_trades.csv", "text/csv", key="trades_tab_download")
