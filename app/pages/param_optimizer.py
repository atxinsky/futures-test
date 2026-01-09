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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)


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
    st.info("期货策略优化功能开发中，请使用命令行版本: `python optuna_optimizer.py`")

    # 显示现有脚本说明
    st.markdown("""
    **现有优化脚本：**
    - `optuna_optimizer.py` - 期货WaveTrend策略优化
    - `optuna_etf_v14.py` - ETF BigBrother V14优化

    **使用方法：**
    ```bash
    cd D:\\期货\\回测改造
    python optuna_optimizer.py
    ```
    """)

    # 显示历史优化结果
    _show_optimization_history("期货")


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
