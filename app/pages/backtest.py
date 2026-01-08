# coding=utf-8
"""
回测系统页面
策略回测配置、执行和结果展示
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def render_backtest_page():
    """
    渲染回测系统页面

    功能：
    1. 策略回测配置和执行
    2. 数据管理入口
    """
    st.title("回测系统")

    # 回测子页面选择
    backtest_page = st.radio(
        "功能选择",
        ["策略回测", "数据管理"],
        horizontal=True,
        label_visibility="collapsed",
        key="backtest_sub_page"
    )

    st.markdown("---")

    if backtest_page == "数据管理":
        # 调用数据管理页面
        from app.pages.data_management import render_data_management_page
        render_data_management_page()
    else:
        # 渲染回测配置和执行
        _render_backtest_main()


def _render_backtest_main():
    """渲染回测主界面"""
    # 导入必要模块
    try:
        from strategies import get_all_strategies, get_strategy
        from config import INSTRUMENTS, get_instrument
        strategies = get_all_strategies()
    except ImportError as e:
        st.error(f"无法加载策略模块: {e}")
        return

    # ============ 三列布局 ============
    col1, col2, col3 = st.columns([1, 1.5, 0.8])

    # 左列：策略和品种选择
    with col1:
        _render_strategy_selection(strategies)

    # 中列：策略参数配置
    with col2:
        _render_strategy_params()

    # 右列：品种规格信息
    with col3:
        _render_instrument_info()

    st.markdown("---")

    # ============ 回测执行按钮 ============
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("开始回测", type="primary", use_container_width=True, key="btn_run_backtest"):
            _run_backtest()

    # ============ 回测结果展示 ============
    if 'backtest_result' in st.session_state and st.session_state.backtest_result:
        st.markdown("---")
        _render_backtest_results()


def _render_strategy_selection(strategies: Dict):
    """渲染策略和品种选择"""
    st.subheader("策略配置")

    # 策略选择
    strategy_names = list(strategies.keys())
    if not strategy_names:
        st.warning("没有可用的策略")
        return

    selected_strategy = st.selectbox(
        "选择策略",
        options=strategy_names,
        key="bt_strategy"
    )

    # 保存到session_state
    st.session_state.selected_strategy = selected_strategy

    # 品种选择
    try:
        from config import INSTRUMENTS
        symbols = list(INSTRUMENTS.keys())
    except:
        symbols = ['RB', 'AU', 'CU', 'I', 'M']

    selected_symbols = st.multiselect(
        "选择品种",
        options=symbols,
        default=['RB'] if 'RB' in symbols else symbols[:1],
        key="bt_symbols"
    )

    st.session_state.selected_symbols = selected_symbols

    # 时间范围
    st.write("**回测时间**")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", key="bt_start")
    with col2:
        end_date = st.date_input("结束日期", key="bt_end")

    st.session_state.bt_start_date = start_date
    st.session_state.bt_end_date = end_date

    # K线周期
    period = st.selectbox(
        "K线周期",
        options=["日线", "60分钟", "30分钟", "15分钟"],
        key="bt_period"
    )
    st.session_state.bt_period = period

    # 初始资金
    capital = st.number_input(
        "初始资金",
        min_value=10000,
        value=100000,
        step=10000,
        key="bt_capital"
    )
    st.session_state.bt_capital = capital


def _render_strategy_params():
    """渲染策略参数配置"""
    st.subheader("策略参数")

    selected_strategy = st.session_state.get('selected_strategy')
    if not selected_strategy:
        st.info("请先选择策略")
        return

    try:
        from strategies import get_strategy
        strategy_class = get_strategy(selected_strategy)

        if strategy_class and hasattr(strategy_class, 'parameters'):
            params = strategy_class.parameters

            # 动态生成参数输入
            param_values = {}
            for param_name, default_value in params.items():
                if isinstance(default_value, bool):
                    param_values[param_name] = st.checkbox(
                        param_name,
                        value=default_value,
                        key=f"param_{param_name}"
                    )
                elif isinstance(default_value, int):
                    param_values[param_name] = st.number_input(
                        param_name,
                        value=default_value,
                        key=f"param_{param_name}"
                    )
                elif isinstance(default_value, float):
                    param_values[param_name] = st.number_input(
                        param_name,
                        value=default_value,
                        format="%.4f",
                        key=f"param_{param_name}"
                    )
                else:
                    param_values[param_name] = st.text_input(
                        param_name,
                        value=str(default_value),
                        key=f"param_{param_name}"
                    )

            st.session_state.bt_params = param_values
        else:
            st.info("该策略没有可配置参数")
            st.session_state.bt_params = {}

    except Exception as e:
        st.error(f"加载策略参数失败: {e}")


def _render_instrument_info():
    """渲染品种规格信息"""
    st.subheader("品种规格")

    selected_symbols = st.session_state.get('selected_symbols', [])
    if not selected_symbols:
        st.info("请选择品种")
        return

    # 显示第一个品种的信息
    symbol = selected_symbols[0]

    try:
        from config import get_instrument
        inst = get_instrument(symbol)

        if inst:
            st.metric("品种代码", symbol)
            st.metric("品种名称", inst.get('name', symbol))
            st.metric("合约乘数", inst.get('multiplier', 0))
            st.metric("保证金率", f"{inst.get('margin_rate', 0)*100:.0f}%")
            st.metric("最小变动", inst.get('tick_size', 0))
        else:
            st.warning(f"未找到品种 {symbol} 的配置")

    except ImportError:
        st.info(f"品种: {symbol}")


def _run_backtest():
    """执行回测"""
    # 获取配置
    strategy_name = st.session_state.get('selected_strategy')
    symbols = st.session_state.get('selected_symbols', [])
    start_date = st.session_state.get('bt_start_date')
    end_date = st.session_state.get('bt_end_date')
    period = st.session_state.get('bt_period', '日线')
    capital = st.session_state.get('bt_capital', 100000)
    params = st.session_state.get('bt_params', {})

    if not strategy_name or not symbols:
        st.error("请选择策略和品种")
        return

    with st.spinner("正在执行回测..."):
        try:
            from engine import BacktestEngine

            # 创建回测引擎
            engine = BacktestEngine(
                strategy_name=strategy_name,
                symbols=symbols,
                start_date=str(start_date),
                end_date=str(end_date),
                initial_capital=capital,
                strategy_params=params
            )

            # 执行回测
            result = engine.run()

            # 保存结果
            st.session_state.backtest_result = result
            st.success("回测完成!")

        except Exception as e:
            st.error(f"回测执行失败: {e}")
            logger.exception("回测执行失败")


def _render_backtest_results():
    """渲染回测结果"""
    result = st.session_state.get('backtest_result')
    if not result:
        return

    st.header("回测结果")

    # 结果Tab
    tab1, tab2, tab3 = st.tabs(["概览", "交易记录", "统计分析"])

    with tab1:
        _render_result_overview(result)

    with tab2:
        _render_trade_records(result)

    with tab3:
        _render_statistics(result)


def _render_result_overview(result):
    """渲染结果概览"""
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("总收益", f"¥{result.get('total_pnl', 0):,.2f}")
    with col2:
        st.metric("收益率", f"{result.get('total_return_pct', 0)*100:.2f}%")
    with col3:
        st.metric("最大回撤", f"{result.get('max_drawdown_pct', 0)*100:.2f}%")
    with col4:
        st.metric("夏普比率", f"{result.get('sharpe_ratio', 0):.2f}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("总交易次数", result.get('total_trades', 0))
    with col2:
        st.metric("胜率", f"{result.get('win_rate', 0)*100:.1f}%")
    with col3:
        st.metric("盈亏比", f"{result.get('profit_factor', 0):.2f}")
    with col4:
        st.metric("平均持仓天数", f"{result.get('avg_holding_days', 0):.1f}")

    # 资金曲线
    equity_curve = result.get('equity_curve')
    if equity_curve is not None and not equity_curve.empty:
        st.subheader("资金曲线")
        st.line_chart(equity_curve['equity'] if 'equity' in equity_curve.columns else equity_curve)


def _render_trade_records(result):
    """渲染交易记录"""
    trades = result.get('trades', [])

    if not trades:
        st.info("暂无交易记录")
        return

    # 转换为DataFrame
    if isinstance(trades[0], dict):
        df = pd.DataFrame(trades)
    else:
        df = pd.DataFrame([t.to_dict() if hasattr(t, 'to_dict') else vars(t) for t in trades])

    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_statistics(result):
    """渲染统计分析"""
    st.subheader("详细统计")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**盈亏统计**")
        st.write(f"- 平均盈利: ¥{result.get('avg_win', 0):,.2f}")
        st.write(f"- 平均亏损: ¥{result.get('avg_loss', 0):,.2f}")
        st.write(f"- 最大单笔盈利: ¥{result.get('max_win', 0):,.2f}")
        st.write(f"- 最大单笔亏损: ¥{result.get('max_loss', 0):,.2f}")

    with col2:
        st.write("**交易统计**")
        st.write(f"- 盈利次数: {result.get('winning_trades', 0)}")
        st.write(f"- 亏损次数: {result.get('losing_trades', 0)}")
        st.write(f"- 总手续费: ¥{result.get('total_commission', 0):,.2f}")
