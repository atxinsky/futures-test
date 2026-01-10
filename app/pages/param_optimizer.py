# coding=utf-8
"""
参数优化页面 - 基于Optuna的策略参数自动优化
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)

# 尝试导入ParamSpaceManager（用于期货策略预定义参数空间）
try:
    from optimization import ParamSpaceManager
    HAS_PARAM_SPACE_MANAGER = True
except ImportError:
    HAS_PARAM_SPACE_MANAGER = False


def render_param_optimizer_page():
    """渲染参数优化页面"""
    st.markdown("### 参数优化")
    st.caption("基于Optuna的智能参数搜索，自动寻找最优策略参数")

    # 检查Optuna是否安装
    try:
        import optuna
        optuna_available = True
    except ImportError:
        optuna_available = False
        st.error("Optuna未安装，请运行: `pip install optuna`")
        return

    # 优化类型选择
    opt_type = st.radio(
        "优化类型",
        ["ETF策略优化", "期货策略优化"],
        horizontal=True,
        key="opt_type"
    )

    st.markdown("---")

    if opt_type == "ETF策略优化":
        _render_etf_optimizer()
    else:
        _render_futures_optimizer()


def _render_etf_optimizer():
    """ETF策略优化界面"""

    # 三列布局
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("#### 优化配置")

        # 策略选择
        strategy_options = [
            "BigBrother V14 (EMA+ADX)",
            "BigBrother V17 (Donchian)",
            "BigBrother V21 (防跳空)"
        ]
        strategy = st.selectbox("选择策略", strategy_options, key="etf_opt_strategy")

        # 时间设置
        st.write("**训练集**")
        train_col1, train_col2 = st.columns(2)
        with train_col1:
            train_start = st.date_input("开始", value=datetime(2021, 1, 1), key="train_start")
        with train_col2:
            train_end = st.date_input("结束", value=datetime(2023, 12, 31), key="train_end")

        st.write("**验证集**")
        val_col1, val_col2 = st.columns(2)
        with val_col1:
            val_start = st.date_input("开始", value=datetime(2024, 1, 1), key="val_start")
        with val_col2:
            val_end = st.date_input("结束", value=datetime.now(), key="val_end")

        # 优化轮数
        n_trials = st.slider("优化轮数", 20, 200, 50, 10, key="n_trials")

        # 优化目标
        opt_target = st.selectbox(
            "优化目标",
            ["sharpe", "calmar", "return", "sortino"],
            format_func=lambda x: {"sharpe": "夏普比率", "calmar": "卡玛比率",
                                   "return": "总收益率", "sortino": "索提诺比率"}[x],
            key="opt_target"
        )

    with col2:
        st.markdown("#### 参数搜索空间")

        if "V14" in strategy:
            # V14参数空间
            st.write("**仓位参数**")
            base_pos_range = st.slider("基础仓位范围", 0.10, 0.30, (0.12, 0.25), 0.02, key="base_pos")
            max_loss_range = st.slider("止损比例范围", 0.03, 0.12, (0.05, 0.10), 0.01, key="max_loss")

            st.write("**止盈参数**")
            trail_start_range = st.slider("追踪触发范围", 0.08, 0.25, (0.10, 0.20), 0.02, key="trail_start")
            trail_stop_range = st.slider("追踪止盈范围", 0.03, 0.10, (0.04, 0.08), 0.01, key="trail_stop")

            st.write("**过滤参数**")
            atr_mult_range = st.slider("ATR倍数范围", 1.5, 4.0, (2.0, 3.5), 0.25, key="atr_mult")
            adx_range = st.slider("ADX阈值范围", 12, 30, (15, 25), 2, key="adx_thresh")

            param_space = {
                'base_position': base_pos_range,
                'max_loss': max_loss_range,
                'trail_start': trail_start_range,
                'trail_stop': trail_stop_range,
                'atr_multiplier': atr_mult_range,
                'adx_threshold': adx_range
            }

        else:
            # V17/V21 Donchian参数空间
            st.write("**风险参数**")
            risk_range = st.slider("单笔风险范围", 0.005, 0.025, (0.008, 0.015), 0.002, key="risk")
            max_pos_range = st.slider("最大仓位范围", 0.15, 0.40, (0.20, 0.35), 0.05, key="max_pos")

            st.write("**通道参数**")
            dc_high_range = st.slider("突破周期范围", 10, 40, (15, 30), 5, key="dc_high")
            dc_low_range = st.slider("跌破周期范围", 5, 25, (8, 15), 2, key="dc_low")

            param_space = {
                'risk_per_trade': risk_range,
                'max_position': max_pos_range,
                'donchian_high_period': dc_high_range,
                'donchian_low_period': dc_low_range
            }

            if "V21" in strategy:
                gap_range = st.slider("高开限制范围", 0.01, 0.05, (0.015, 0.03), 0.005, key="gap")
                param_space['gap_up_limit'] = gap_range

    with col3:
        st.markdown("#### 标的池")

        from core.etf_data_service import ETF_POOLS, ALL_ETFS, BIGBROTHER_POOL

        pool_options = ["默认池"] + list(ETF_POOLS.keys())
        selected_pool = st.selectbox("预设池", pool_options, key="opt_pool")

        if selected_pool == "默认池":
            default_codes = BIGBROTHER_POOL
        else:
            default_codes = list(ETF_POOLS[selected_pool].keys())

        etf_pool = st.multiselect(
            "选择ETF",
            options=list(ALL_ETFS.keys()),
            default=default_codes[:6],
            format_func=lambda x: f"{x} - {ALL_ETFS.get(x, '')}",
            key="opt_etf_pool"
        )

        st.markdown("---")

        # 高级选项
        with st.expander("高级选项"):
            initial_capital = st.number_input("初始资金", 100000, 10000000, 1000000, 100000)
            min_trades = st.number_input("最少交易次数", 5, 50, 15, 5)
            max_drawdown = st.slider("最大回撤限制", 0.20, 0.50, 0.35, 0.05)

    st.markdown("---")

    # 运行按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_btn = st.button("🚀 开始优化", type="primary", use_container_width=True, key="run_opt")

    if run_btn:
        if not etf_pool:
            st.error("请至少选择一个ETF")
            return

        _run_etf_optimization(
            strategy=strategy,
            train_start=train_start.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            val_start=val_start.strftime("%Y-%m-%d"),
            val_end=val_end.strftime("%Y-%m-%d"),
            n_trials=n_trials,
            opt_target=opt_target,
            param_space=param_space,
            etf_pool=etf_pool,
            initial_capital=initial_capital,
            min_trades=min_trades,
            max_drawdown=max_drawdown
        )

    # 显示历史优化结果
    _show_optimization_history("ETF")


def _render_futures_optimizer():
    """期货策略优化界面"""

    # 三列布局
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("#### 优化配置")

        # 策略选择
        strategy_options = {
            "Brother2v6 (趋势突破)": "brother2v6",
            "WaveTrend Final": "wavetrend_final",
            "EMANew V5": "emanew_v5",
            "Donchian Trend": "donchian_trend",
            "Dual MA": "dual_ma",
        }
        strategy_display = st.selectbox("选择策略", list(strategy_options.keys()), key="futures_opt_strategy")
        strategy_key = strategy_options[strategy_display]

        # 品种选择
        from config import INSTRUMENTS
        symbols = list(INSTRUMENTS.keys())
        default_symbols = ["RB", "I", "MA", "TA", "IF"]
        default_symbols = [s for s in default_symbols if s in symbols]

        selected_symbols = st.multiselect(
            "选择品种",
            options=symbols,
            default=default_symbols[:3],
            format_func=lambda x: f"{x} - {INSTRUMENTS[x]['name']}",
            key="futures_opt_symbols"
        )

        # 时间设置
        st.write("**训练集**")
        train_col1, train_col2 = st.columns(2)
        with train_col1:
            train_start = st.date_input("开始", value=datetime(2019, 1, 1), key="fut_train_start")
        with train_col2:
            train_end = st.date_input("结束", value=datetime(2023, 12, 31), key="fut_train_end")

        st.write("**验证集**")
        val_col1, val_col2 = st.columns(2)
        with val_col1:
            val_start = st.date_input("开始", value=datetime(2024, 1, 1), key="fut_val_start")
        with val_col2:
            val_end = st.date_input("结束", value=datetime.now(), key="fut_val_end")

        # 优化轮数
        n_trials = st.slider("优化轮数", 20, 200, 50, 10, key="fut_n_trials")

        # 优化目标
        opt_target = st.selectbox(
            "优化目标",
            ["sharpe", "calmar", "return", "sortino"],
            format_func=lambda x: {"sharpe": "夏普比率", "calmar": "卡玛比率",
                                   "return": "总收益率", "sortino": "索提诺比率"}[x],
            key="fut_opt_target"
        )

    with col2:
        st.markdown("#### 参数搜索空间")
        param_space = _get_futures_param_space(strategy_key)

    with col3:
        st.markdown("#### 高级设置")

        initial_capital = st.number_input("初始资金", 50000, 1000000, 100000, 10000, key="fut_capital")
        min_trades = st.number_input("最少交易次数", 1, 50, 5, 1, key="fut_min_trades")
        max_drawdown = st.slider("最大回撤限制", 0.20, 0.60, 0.40, 0.05, key="fut_max_dd")

        st.markdown("---")
        st.caption("**提示：** 期货优化可能较慢，建议先用少量品种测试")

    st.markdown("---")

    # 运行按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_btn = st.button("🚀 开始优化", type="primary", use_container_width=True, key="fut_run_opt")

    if run_btn:
        if not selected_symbols:
            st.error("请至少选择一个品种")
            return

        _run_futures_optimization(
            strategy_key=strategy_key,
            strategy_display=strategy_display,
            symbols=selected_symbols,
            train_start=train_start.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            val_start=val_start.strftime("%Y-%m-%d"),
            val_end=val_end.strftime("%Y-%m-%d"),
            n_trials=n_trials,
            opt_target=opt_target,
            param_space=param_space,
            initial_capital=initial_capital,
            min_trades=min_trades,
            max_drawdown=max_drawdown
        )

    # 显示历史优化结果
    _show_optimization_history("期货")


def _get_futures_param_space(strategy_key: str) -> dict:
    """根据策略生成参数搜索空间UI"""
    param_space = {}

    # 尝试从ParamSpaceManager获取预定义参数空间
    predefined_space = None
    if HAS_PARAM_SPACE_MANAGER and strategy_key in ParamSpaceManager.get_supported_strategies():
        predefined_space = ParamSpaceManager.get_param_space(strategy_key)
        st.success(f"已加载 {strategy_key} 预定义参数空间（{len(predefined_space)}个参数）")

    if strategy_key == "brother2v6":
        # 使用预定义空间的值作为默认范围
        if predefined_space:
            st.write("**趋势参数**")
            ps = {p.name: p for p in predefined_space}
            sml_len = st.slider("短期EMA范围",
                int(ps['sml_len'].low), int(ps['sml_len'].high),
                (int(ps['sml_len'].low), int(ps['sml_len'].high)), 1, key="b6_sml")
            big_len = st.slider("长期EMA范围",
                int(ps['big_len'].low), int(ps['big_len'].high),
                (int(ps['big_len'].low), int(ps['big_len'].high)), 5, key="b6_big")
            break_len = st.slider("突破周期范围",
                int(ps['break_len'].low), int(ps['break_len'].high),
                (int(ps['break_len'].low), int(ps['break_len'].high)), 5, key="b6_break")

            st.write("**过滤参数**")
            adx_thres = st.slider("ADX阈值范围",
                ps['adx_thres'].low, ps['adx_thres'].high,
                (ps['adx_thres'].low, ps['adx_thres'].high), 1.0, key="b6_adx")
            chop_thres = st.slider("CHOP阈值范围",
                ps['chop_thres'].low, ps['chop_thres'].high,
                (ps['chop_thres'].low, ps['chop_thres'].high), 1.0, key="b6_chop")
            vol_multi = st.slider("放量倍数范围",
                ps['vol_multi'].low, ps['vol_multi'].high,
                (ps['vol_multi'].low, ps['vol_multi'].high), 0.1, key="b6_vol")

            st.write("**止损参数**")
            stop_n = st.slider("止损ATR倍数",
                ps['stop_n'].low, ps['stop_n'].high,
                (ps['stop_n'].low, ps['stop_n'].high), 0.5, key="b6_stop")
        else:
            st.write("**趋势参数**")
            sml_len = st.slider("短期EMA范围", 8, 18, (10, 15), 1, key="b6_sml")
            big_len = st.slider("长期EMA范围", 35, 70, (45, 55), 5, key="b6_big")
            break_len = st.slider("突破周期范围", 20, 45, (25, 35), 5, key="b6_break")

            st.write("**过滤参数**")
            adx_thres = st.slider("ADX阈值范围", 18.0, 28.0, (20.0, 25.0), 1.0, key="b6_adx")
            chop_thres = st.slider("CHOP阈值范围", 45.0, 55.0, (48.0, 52.0), 1.0, key="b6_chop")
            vol_multi = st.slider("放量倍数范围", 1.1, 2.0, (1.2, 1.5), 0.1, key="b6_vol")

            st.write("**止损参数**")
            stop_n = st.slider("止损ATR倍数", 2.0, 4.5, (2.5, 3.5), 0.5, key="b6_stop")

        param_space = {
            'sml_len': sml_len, 'big_len': big_len, 'break_len': break_len,
            'adx_thres': adx_thres, 'chop_thres': chop_thres, 'vol_multi': vol_multi,
            'stop_n': stop_n
        }

    elif strategy_key == "wavetrend_final":
        st.write("**WaveTrend参数**")
        n1 = st.slider("通道长度范围", 5, 20, (8, 15), 1, key="wt_n1")
        n2 = st.slider("平均长度范围", 10, 30, (15, 25), 1, key="wt_n2")
        ob_level = st.slider("超买阈值范围", 40, 70, (50, 65), 5, key="wt_ob")
        os_level = st.slider("超卖阈值范围", -70, -40, (-60, -45), 5, key="wt_os")

        st.write("**止损参数**")
        atr_mult = st.slider("ATR倍数范围", 1.5, 5.0, (2.0, 3.5), 0.5, key="wt_atr")

        param_space = {
            'n1': n1, 'n2': n2, 'ob_level': ob_level, 'os_level': os_level,
            'atr_mult': atr_mult
        }

    elif strategy_key == "emanew_v5":
        st.write("**EMA参数**")
        fast_len = st.slider("快线周期范围", 5, 15, (8, 12), 1, key="ema_fast")
        slow_len = st.slider("慢线周期范围", 20, 50, (25, 40), 5, key="ema_slow")

        st.write("**过滤参数**")
        adx_thres = st.slider("ADX阈值范围", 15.0, 30.0, (18.0, 25.0), 1.0, key="ema_adx")

        st.write("**止损参数**")
        atr_mult = st.slider("ATR倍数范围", 1.5, 4.0, (2.0, 3.0), 0.5, key="ema_atr")

        param_space = {
            'fast_len': fast_len, 'slow_len': slow_len,
            'adx_thres': adx_thres, 'atr_mult': atr_mult
        }

    elif strategy_key == "donchian_trend":
        st.write("**通道参数**")
        high_period = st.slider("突破周期范围", 10, 40, (15, 30), 5, key="dc_high")
        low_period = st.slider("跌破周期范围", 5, 25, (8, 15), 2, key="dc_low")

        st.write("**止损参数**")
        atr_mult = st.slider("ATR倍数范围", 1.5, 4.0, (2.0, 3.0), 0.5, key="dc_atr")

        param_space = {
            'high_period': high_period, 'low_period': low_period,
            'atr_mult': atr_mult
        }

    elif strategy_key == "dual_ma":
        st.write("**均线参数**")
        fast_period = st.slider("快线周期范围", 5, 20, (8, 15), 1, key="ma_fast")
        slow_period = st.slider("慢线周期范围", 20, 60, (30, 50), 5, key="ma_slow")

        st.write("**止损参数**")
        stop_pct = st.slider("止损比例范围(%)", 2.0, 8.0, (3.0, 6.0), 0.5, key="ma_stop")

        param_space = {
            'fast_period': fast_period, 'slow_period': slow_period,
            'stop_pct': stop_pct
        }

    else:
        st.warning("该策略暂未配置参数空间")

    return param_space


def _run_futures_optimization(strategy_key, strategy_display, symbols, train_start, train_end,
                               val_start, val_end, n_trials, opt_target, param_space,
                               initial_capital, min_trades, max_drawdown):
    """运行期货参数优化"""
    import optuna

    # 进度显示
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.empty()

    logs = []

    def log(msg):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        log_container.code("\n".join(logs[-10:]))

    log(f"开始优化: {strategy_display}")
    log(f"品种: {', '.join(symbols)}")
    log(f"训练集: {train_start} ~ {train_end}")
    log(f"优化轮数: {n_trials}")

    # 加载数据
    status_text.text("加载数据...")
    try:
        from core.backtest_engine import BacktestEngine
        from utils.data_loader import load_futures_data

        all_data = {}
        for i, symbol in enumerate(symbols):
            log(f"加载 {symbol} ({i+1}/{len(symbols)})...")
            status_text.text(f"加载数据: {symbol} ({i+1}/{len(symbols)})")

            df = load_futures_data(symbol, train_start, val_end, auto_download=True)
            if df is not None and len(df) > 0:
                all_data[symbol] = df
                log(f"  {symbol}: {len(df)}行")
            else:
                log(f"  {symbol}: 无数据，跳过")

        if not all_data:
            st.error("无法加载任何品种数据。请检查：\n1. 天勤账号是否配置正确\n2. 网络是否正常\n3. 数据库路径是否正确")
            st.info("数据库路径: D:\\期货\\回测改造\\data\\futures_tq.db")
            return

        log(f"数据加载完成，共 {len(all_data)} 个品种")

    except Exception as e:
        st.error(f"数据加载失败: {e}")
        logger.exception("数据加载失败")
        return

    # 获取策略类
    try:
        strategy_class = _get_strategy_class(strategy_key)
        if strategy_class is None:
            st.error(f"无法加载策略: {strategy_key}")
            return
    except Exception as e:
        st.error(f"策略加载失败: {e}")
        return

    # 定义目标函数
    trial_results = []

    def objective(trial):
        # 构建参数
        params = {}
        for param_name, (low, high) in param_space.items():
            if isinstance(low, int):
                params[param_name] = trial.suggest_int(param_name, low, high)
            else:
                step = round((high - low) / 10, 2)
                if step < 0.01:
                    step = 0.01
                params[param_name] = trial.suggest_float(param_name, low, high, step=step)

        # 多品种综合回测
        total_sharpe = 0
        total_return = 0
        total_trades = 0
        max_dd = 0
        valid_count = 0

        for symbol, df in all_data.items():
            try:
                # 筛选训练集时间范围
                train_df = df[(df.index >= train_start) & (df.index <= train_end)]
                if len(train_df) < 100:
                    continue

                strategy = strategy_class(params=params)
                engine = BacktestEngine()
                result = engine.run(
                    strategy=strategy,
                    symbol=symbol,
                    data=train_df,
                    initial_capital=initial_capital,
                    check_limit_price=False
                )

                if result and result.total_trades > 0:
                    total_sharpe += result.sharpe_ratio or 0
                    total_return += result.total_return or 0
                    total_trades += result.total_trades or 0
                    max_dd = max(max_dd, result.max_drawdown or 0)
                    valid_count += 1

            except Exception as e:
                logger.warning(f"回测 {symbol} 失败: {e}")
                continue

        if valid_count == 0:
            return -999

        avg_sharpe = total_sharpe / valid_count
        avg_return = total_return / valid_count

        # 惩罚条件
        if total_trades < min_trades:
            return -999
        if max_dd > max_drawdown:
            return -999

        # 记录结果
        trial_results.append({
            'trial': trial.number,
            'params': params.copy(),
            'sharpe': avg_sharpe,
            'return': avg_return,
            'drawdown': max_dd,
            'trades': total_trades
        })

        # 返回目标值
        if opt_target == 'sharpe':
            return avg_sharpe
        elif opt_target == 'calmar':
            return avg_return / max_dd if max_dd > 0 else avg_return
        elif opt_target == 'return':
            return avg_return
        else:
            return avg_sharpe

    # 创建Study
    status_text.text("创建优化器...")
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # 运行优化
    status_text.text("开始优化...")

    def callback(study, trial):
        progress = (trial.number + 1) / n_trials
        progress_bar.progress(progress)
        if trial.value and trial.value > -900:
            log(f"Trial {trial.number}: {opt_target}={trial.value:.3f}")

    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
    except Exception as e:
        st.error(f"优化失败: {e}")
        logger.exception("优化失败")
        return

    progress_bar.progress(1.0)
    status_text.text("优化完成!")
    log("优化完成!")

    # 获取最优参数
    best_params = study.best_params
    best_value = study.best_value

    st.success(f"最优{opt_target}: {best_value:.3f}")

    # 显示最优参数
    st.markdown("#### 最优参数")
    params_df = pd.DataFrame([
        {"参数": k, "最优值": f"{v:.4f}" if isinstance(v, float) else str(v)}
        for k, v in best_params.items()
    ])
    st.dataframe(params_df, hide_index=True, use_container_width=True)

    # 验证集测试
    st.markdown("#### 验证集测试")
    _validate_futures_params(strategy_class, best_params, all_data, train_start, train_end, val_start, val_end, initial_capital)

    # 参数重要性
    st.markdown("#### 参数重要性")
    try:
        importances = optuna.importance.get_param_importances(study)
        imp_df = pd.DataFrame([
            {"参数": k, "重要性": v}
            for k, v in sorted(importances.items(), key=lambda x: -x[1])
        ])

        fig = go.Figure(go.Bar(
            x=imp_df['重要性'],
            y=imp_df['参数'],
            orientation='h',
            marker_color='#1f77b4'
        ))
        fig.update_layout(height=300, margin=dict(l=100, r=50, t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"无法计算参数重要性: {e}")

    # 优化过程图
    st.markdown("#### 优化收敛过程")
    if trial_results:
        results_df = pd.DataFrame(trial_results)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['trial'],
            y=results_df['sharpe'],
            mode='markers+lines',
            name='Sharpe',
            marker=dict(size=6)
        ))
        cummax = results_df['sharpe'].cummax()
        fig.add_trace(go.Scatter(
            x=results_df['trial'],
            y=cummax,
            mode='lines',
            name='累计最优',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(height=300, xaxis_title='Trial', yaxis_title='Sharpe')
        st.plotly_chart(fig, use_container_width=True)

    # 保存结果
    _save_optimization_result(
        opt_type="期货",
        strategy=strategy_display,
        best_params=best_params,
        best_value=best_value,
        opt_target=opt_target,
        n_trials=n_trials,
        train_range=f"{train_start}~{train_end}",
        val_range=f"{val_start}~{val_end}"
    )


def _get_strategy_class(strategy_key: str):
    """根据key获取策略类"""
    try:
        if strategy_key == "brother2v6":
            from strategies.brother2v6 import Brother2v6Strategy
            return Brother2v6Strategy
        elif strategy_key == "wavetrend_final":
            from strategies.wavetrend_final import WaveTrendFinalStrategy
            return WaveTrendFinalStrategy
        elif strategy_key == "emanew_v5":
            from strategies.emanew_v5 import EMANewV5Strategy
            return EMANewV5Strategy
        elif strategy_key == "donchian_trend":
            from strategies.donchian_trend import DonchianTrendStrategy
            return DonchianTrendStrategy
        elif strategy_key == "dual_ma":
            from strategies.dual_ma import DualMAStrategy
            return DualMAStrategy
        else:
            return None
    except ImportError as e:
        logger.warning(f"策略导入失败: {e}")
        return None


def _validate_futures_params(strategy_class, params, all_data, train_start, train_end, val_start, val_end, initial_capital):
    """验证集测试"""
    from core.backtest_engine import BacktestEngine

    results = {}

    for period_name, start, end in [("训练集", train_start, train_end), ("验证集", val_start, val_end)]:
        total_sharpe = 0
        total_return = 0
        max_dd = 0
        total_trades = 0
        valid_count = 0

        for symbol, df in all_data.items():
            try:
                period_df = df[(df.index >= start) & (df.index <= end)]
                if len(period_df) < 50:
                    continue

                strategy = strategy_class(params=params)
                engine = BacktestEngine()
                result = engine.run(
                    strategy=strategy,
                    symbol=symbol,
                    data=period_df,
                    initial_capital=initial_capital
                )

                if result:
                    total_sharpe += result.sharpe_ratio or 0
                    total_return += result.total_return or 0
                    max_dd = max(max_dd, result.max_drawdown or 0)
                    total_trades += result.total_trades or 0
                    valid_count += 1

            except Exception as e:
                logger.warning(f"{period_name} {symbol} 测试失败: {e}")

        if valid_count > 0:
            results[period_name] = {
                'sharpe': total_sharpe / valid_count,
                'return': total_return / valid_count,
                'drawdown': max_dd,
                'trades': total_trades
            }

    if results.get("训练集") and results.get("验证集"):
        train = results["训练集"]
        val = results["验证集"]

        decay = (train['sharpe'] - val['sharpe']) / train['sharpe'] * 100 if train['sharpe'] > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("训练集Sharpe", f"{train['sharpe']:.3f}")
            st.metric("训练集收益", f"{train['return']*100:.1f}%")
        with col2:
            st.metric("验证集Sharpe", f"{val['sharpe']:.3f}")
            st.metric("验证集收益", f"{val['return']*100:.1f}%")
        with col3:
            if decay > 40:
                st.error(f"衰减: {decay:.1f}% (过拟合风险高)")
            elif decay > 20:
                st.warning(f"衰减: {decay:.1f}% (轻度过拟合)")
            else:
                st.success(f"衰减: {decay:.1f}% (参数稳健)")


def _run_etf_optimization(strategy, train_start, train_end, val_start, val_end,
                          n_trials, opt_target, param_space, etf_pool,
                          initial_capital, min_trades, max_drawdown):
    """运行ETF参数优化"""
    import optuna

    # 进度显示
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.empty()

    logs = []

    def log(msg):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        log_container.code("\n".join(logs[-10:]))

    log(f"开始优化: {strategy}")
    log(f"训练集: {train_start} ~ {train_end}")
    log(f"验证集: {val_start} ~ {val_end}")
    log(f"优化轮数: {n_trials}")

    # 加载数据
    status_text.text("加载数据...")
    try:
        from core.etf_data_service import get_etf_data_service
        ds = get_etf_data_service()

        data = {}
        for code in etf_pool:
            df = ds.get_data_with_indicators(code, train_start, val_end)
            if len(df) > 0:
                data[code] = df
                log(f"加载: {code} - {len(df)}行")

        if not data:
            st.error("无法加载数据")
            return

    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return

    # 定义目标函数
    trial_results = []

    def objective(trial):
        # 构建参数
        params = {}
        for param_name, (low, high) in param_space.items():
            if isinstance(low, int):
                params[param_name] = trial.suggest_int(param_name, low, high)
            else:
                step = (high - low) / 10
                params[param_name] = trial.suggest_float(param_name, low, high, step=step)

        # 运行回测
        try:
            from core.etf_backtest_engine import ETFBacktestEngine

            if "V14" in strategy:
                from strategies.etf_bigbrother_v14 import ETFBigBrotherV14
                strat = ETFBigBrotherV14(pool=etf_pool, **params)
            elif "V17" in strategy:
                from strategies.etf_bigbrother_v17_v21 import ETFBigBrotherV17
                strat = ETFBigBrotherV17(pool=etf_pool, **params)
            elif "V21" in strategy:
                from strategies.etf_bigbrother_v17_v21 import ETFBigBrotherV21
                strat = ETFBigBrotherV21(pool=etf_pool, **params)
            else:
                from strategies.etf_bigbrother_v14 import ETFBigBrotherV14
                strat = ETFBigBrotherV14(pool=etf_pool, **params)

            engine = ETFBacktestEngine(
                initial_capital=initial_capital,
                commission_rate=0.0001
            )
            engine.set_strategy(strat.initialize, strat.handle_data)

            result = engine.run(data=data, start_date=train_start, end_date=train_end)

            # 惩罚条件
            if result.total_trades < min_trades:
                return -999
            if result.max_drawdown > max_drawdown:
                return -999

            # 记录结果
            trial_results.append({
                'trial': trial.number,
                'params': params.copy(),
                'sharpe': result.sharpe_ratio,
                'return': result.total_return,
                'drawdown': result.max_drawdown,
                'trades': result.total_trades
            })

            # 返回目标值
            if opt_target == 'sharpe':
                return result.sharpe_ratio or 0
            elif opt_target == 'calmar':
                return result.calmar_ratio or 0
            elif opt_target == 'return':
                return result.total_return or 0
            else:
                return result.sharpe_ratio or 0

        except Exception as e:
            logger.warning(f"Trial {trial.number} 失败: {e}")
            return -999

    # 创建Study
    status_text.text("创建优化器...")
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # 运行优化
    status_text.text("开始优化...")

    def callback(study, trial):
        progress = (trial.number + 1) / n_trials
        progress_bar.progress(progress)
        if trial.value and trial.value > -900:
            log(f"Trial {trial.number}: {opt_target}={trial.value:.3f}")

    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
    except Exception as e:
        st.error(f"优化失败: {e}")
        return

    progress_bar.progress(1.0)
    status_text.text("优化完成!")
    log("优化完成!")

    # 获取最优参数
    best_params = study.best_params
    best_value = study.best_value

    st.success(f"最优{opt_target}: {best_value:.3f}")

    # 显示最优参数
    st.markdown("#### 最优参数")
    params_df = pd.DataFrame([
        {"参数": k, "最优值": f"{v:.4f}" if isinstance(v, float) else str(v)}
        for k, v in best_params.items()
    ])
    st.dataframe(params_df, hide_index=True, use_container_width=True)

    # 验证集测试
    st.markdown("#### 验证集测试")
    _validate_params(strategy, best_params, data, train_start, train_end, val_start, val_end, etf_pool, initial_capital)

    # 参数重要性
    st.markdown("#### 参数重要性")
    try:
        importances = optuna.importance.get_param_importances(study)
        imp_df = pd.DataFrame([
            {"参数": k, "重要性": v}
            for k, v in sorted(importances.items(), key=lambda x: -x[1])
        ])

        fig = go.Figure(go.Bar(
            x=imp_df['重要性'],
            y=imp_df['参数'],
            orientation='h',
            marker_color='#1f77b4'
        ))
        fig.update_layout(height=300, margin=dict(l=100, r=50, t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"无法计算参数重要性: {e}")

    # 优化过程图
    st.markdown("#### 优化收敛过程")
    if trial_results:
        results_df = pd.DataFrame(trial_results)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['trial'],
            y=results_df['sharpe'],
            mode='markers+lines',
            name='Sharpe',
            marker=dict(size=6)
        ))
        # 累计最优
        cummax = results_df['sharpe'].cummax()
        fig.add_trace(go.Scatter(
            x=results_df['trial'],
            y=cummax,
            mode='lines',
            name='累计最优',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(height=300, xaxis_title='Trial', yaxis_title='Sharpe')
        st.plotly_chart(fig, use_container_width=True)

    # 保存结果
    _save_optimization_result(
        opt_type="ETF",
        strategy=strategy,
        best_params=best_params,
        best_value=best_value,
        opt_target=opt_target,
        n_trials=n_trials,
        train_range=f"{train_start}~{train_end}",
        val_range=f"{val_start}~{val_end}"
    )

    # 生成可复制代码
    st.markdown("#### 复制代码")
    code = _generate_strategy_code(strategy, best_params, etf_pool)
    st.code(code, language='python')

    # 一键应用到回测
    st.markdown("---")
    st.markdown("#### 应用到回测")
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("应用参数到ETF回测页面", type="primary", use_container_width=True, key="apply_to_backtest"):
            # 保存到session_state
            st.session_state['opt_apply_params'] = {
                'strategy': strategy,
                'params': best_params,
                'etf_pool': etf_pool,
                'train_range': f"{train_start}~{train_end}",
                'val_range': f"{val_start}~{val_end}",
                'best_value': best_value,
                'opt_target': opt_target
            }
            st.success("参数已保存！请切换到 ETF回测 页面")
            st.balloons()
    with col2:
        st.caption("点击后前往侧边栏 ETF回测 页面应用")


def _validate_params(strategy, params, data, train_start, train_end, val_start, val_end, etf_pool, initial_capital):
    """验证集测试"""
    from core.etf_backtest_engine import ETFBacktestEngine

    results = {}

    for period_name, start, end in [("训练集", train_start, train_end), ("验证集", val_start, val_end)]:
        try:
            if "V14" in strategy:
                from strategies.etf_bigbrother_v14 import ETFBigBrotherV14
                strat = ETFBigBrotherV14(pool=etf_pool, **params)
            elif "V17" in strategy:
                from strategies.etf_bigbrother_v17_v21 import ETFBigBrotherV17
                strat = ETFBigBrotherV17(pool=etf_pool, **params)
            elif "V21" in strategy:
                from strategies.etf_bigbrother_v17_v21 import ETFBigBrotherV21
                strat = ETFBigBrotherV21(pool=etf_pool, **params)
            else:
                from strategies.etf_bigbrother_v14 import ETFBigBrotherV14
                strat = ETFBigBrotherV14(pool=etf_pool, **params)

            engine = ETFBacktestEngine(initial_capital=initial_capital, commission_rate=0.0001)
            engine.set_strategy(strat.initialize, strat.handle_data)
            result = engine.run(data=data, start_date=start, end_date=end)

            results[period_name] = {
                'sharpe': result.sharpe_ratio,
                'return': result.total_return,
                'drawdown': result.max_drawdown,
                'trades': result.total_trades,
                'win_rate': result.win_rate
            }
        except Exception as e:
            results[period_name] = None
            logger.warning(f"{period_name}测试失败: {e}")

    if results.get("训练集") and results.get("验证集"):
        train = results["训练集"]
        val = results["验证集"]

        decay = (train['sharpe'] - val['sharpe']) / train['sharpe'] * 100 if train['sharpe'] > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("训练集Sharpe", f"{train['sharpe']:.3f}")
            st.metric("训练集收益", f"{train['return']*100:.1f}%")
        with col2:
            st.metric("验证集Sharpe", f"{val['sharpe']:.3f}")
            st.metric("验证集收益", f"{val['return']*100:.1f}%")
        with col3:
            if decay > 40:
                st.error(f"衰减: {decay:.1f}% (过拟合风险高)")
            elif decay > 20:
                st.warning(f"衰减: {decay:.1f}% (轻度过拟合)")
            else:
                st.success(f"衰减: {decay:.1f}% (参数稳健)")


def _save_optimization_result(opt_type, strategy, best_params, best_value, opt_target, n_trials, train_range, val_range):
    """保存优化结果"""
    try:
        from utils.backtest_storage import get_backtest_storage
        import json

        storage = get_backtest_storage()

        # 保存为特殊的回测记录（strategy_name前缀加[OPT]）
        from utils.backtest_storage import BacktestRecord
        import sqlite3

        record = BacktestRecord(
            backtest_id=f"OPT_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            backtest_type=opt_type,
            strategy_name=f"[优化] {strategy}",
            symbols="",
            start_date=train_range.split("~")[0],
            end_date=train_range.split("~")[1],
            initial_capital=0,
            final_value=0,
            total_return=0,
            annual_return=0,
            max_drawdown=0,
            sharpe_ratio=best_value,
            win_rate=0,
            total_trades=n_trials,
            params_json=json.dumps(best_params, ensure_ascii=False),
            result_json=json.dumps({
                'opt_target': opt_target,
                'best_value': best_value,
                'n_trials': n_trials,
                'train_range': train_range,
                'val_range': val_range
            }, ensure_ascii=False),
            trades_json="[]",
            equity_csv="",
            notes=f"Optuna优化结果 | 目标:{opt_target}={best_value:.3f}",
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        storage._save_record(record)
        st.success("优化结果已保存到回测历史")

    except Exception as e:
        logger.warning(f"保存优化结果失败: {e}")


def _show_optimization_history(opt_type):
    """显示历史优化结果"""
    st.markdown("---")
    st.markdown("#### 历史优化记录")

    try:
        from utils.backtest_storage import get_backtest_storage
        storage = get_backtest_storage()

        # 获取优化记录（strategy_name以[优化]开头）
        records = storage.get_records(backtest_type=opt_type, limit=50)
        opt_records = [r for r in records if r.strategy_name.startswith("[优化]")]

        if not opt_records:
            st.info("暂无优化记录")
            return

        # 显示列表
        data = []
        for r in opt_records[:10]:
            params = json.loads(r.params_json) if r.params_json else {}
            result = json.loads(r.result_json) if r.result_json else {}

            data.append({
                "时间": r.created_at[:16],
                "策略": r.strategy_name.replace("[优化] ", ""),
                "目标": result.get('opt_target', '-'),
                "最优值": f"{result.get('best_value', 0):.3f}",
                "轮数": result.get('n_trials', 0),
                "备注": r.notes[:30] if r.notes else ""
            })

        df = pd.DataFrame(data)
        st.dataframe(df, hide_index=True, use_container_width=True)

    except Exception as e:
        st.warning(f"无法加载历史记录: {e}")


def _generate_strategy_code(strategy, params, etf_pool):
    """生成策略代码"""
    pool_str = ",\n    ".join([f'"{c}"' for c in etf_pool])

    if "V14" in strategy:
        return f'''# BigBrother V14 最优参数
from strategies.etf_bigbrother_v14 import ETFBigBrotherV14

ETF_POOL = [
    {pool_str}
]

strategy = ETFBigBrotherV14(
    pool=ETF_POOL,
    base_position={params.get('base_position', 0.18):.4f},
    atr_multiplier={params.get('atr_multiplier', 2.5):.2f},
    max_loss={params.get('max_loss', 0.07):.4f},
    trail_start={params.get('trail_start', 0.15):.4f},
    trail_stop={params.get('trail_stop', 0.06):.4f},
    adx_threshold={params.get('adx_threshold', 20)},
)'''

    elif "V17" in strategy or "V21" in strategy:
        class_name = "ETFBigBrotherV21" if "V21" in strategy else "ETFBigBrotherV17"
        code = f'''# {strategy} 最优参数
from strategies.etf_bigbrother_v17_v21 import {class_name}

ETF_POOL = [
    {pool_str}
]

strategy = {class_name}(
    pool=ETF_POOL,
    risk_per_trade={params.get('risk_per_trade', 0.01):.4f},
    max_position={params.get('max_position', 0.25):.4f},
    donchian_high_period={params.get('donchian_high_period', 20)},
    donchian_low_period={params.get('donchian_low_period', 10)},'''

        if "V21" in strategy and 'gap_up_limit' in params:
            code += f'''
    gap_up_limit={params.get('gap_up_limit', 0.02):.4f},'''

        code += '''
)'''
        return code

    return "# 参数代码生成失败"
