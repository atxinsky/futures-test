# coding=utf-8
"""
AI自动参数优化页面
一键全参数优化，支持每品种独立优化
"""

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
import plotly.graph_objects as go

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

# 过拟合检测阈值
OVERFITTING_THRESHOLD_HIGH = 40    # 严重过拟合
OVERFITTING_THRESHOLD_MEDIUM = 20  # 轻度过拟合


def render_auto_optimizer_page():
    """渲染AI自动优化页面"""
    st.markdown("### AI参数优化")
    st.caption("一键全参数优化，自动保存最优配置到YAML文件")

    # 检查Optuna是否安装
    try:
        import optuna
        optuna_available = True
    except ImportError:
        optuna_available = False
        st.error("Optuna未安装，请运行: `pip install optuna`")
        st.code("pip install optuna")
        return

    # 导入优化模块
    try:
        from optimization import OptunaOptimizer, OptimizationConfig, ParamSpaceManager, ConfigApplier
    except ImportError as e:
        st.error(f"优化模块导入失败: {e}")
        st.info("请确保 optimization 目录存在且包含所有必要文件")
        return

    # 导入配置
    try:
        from config import INSTRUMENTS
    except ImportError:
        INSTRUMENTS = {
            "RB": {"name": "螺纹钢"}, "I": {"name": "铁矿石"},
            "MA": {"name": "甲醇"}, "IF": {"name": "沪深300"},
            "AU": {"name": "黄金"}, "CU": {"name": "铜"},
        }

    # 三列布局
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("#### 策略配置")

        # 获取支持的策略列表
        supported_strategies = ParamSpaceManager.get_supported_strategies()

        strategy_display_names = {
            "brother2v6": "Brother2v6 (趋势突破)",
            "brother2v5": "Brother2v5 (经典版)",
            "brother2_enhanced": "Brother2 Enhanced",
            "brother2v6_dual": "Brother2v6 Dual (双向)",
        }

        strategy_options = {
            strategy_display_names.get(s, s): s
            for s in supported_strategies
        }

        if not strategy_options:
            st.warning("未找到支持的策略")
            return

        strategy_display = st.selectbox(
            "选择策略",
            list(strategy_options.keys()),
            key="auto_opt_strategy"
        )
        strategy_name = strategy_options[strategy_display]

        # 品种选择
        symbols = list(INSTRUMENTS.keys())
        default_symbols = ["RB", "I", "MA", "IF"]
        selected_symbols = st.multiselect(
            "选择品种",
            options=symbols,
            default=[s for s in default_symbols if s in symbols][:3],
            format_func=lambda x: f"{x} - {INSTRUMENTS[x].get('name', x)}",
            key="auto_opt_symbols"
        )

        # 优化模式
        opt_mode = st.radio(
            "优化模式",
            ["多品种综合优化", "每品种独立优化"],
            help="综合优化：所有品种共享参数；独立优化：每品种单独优化参数",
            key="auto_opt_mode"
        )

    with col2:
        st.markdown("#### 参数配置")

        # 参数集合选择
        all_params = ParamSpaceManager.get_all_params(strategy_name)
        key_params = ParamSpaceManager.get_key_params(strategy_name)

        param_preset = st.radio(
            "参数集合",
            [f"全参数 ({len(all_params)}个)", f"关键参数 ({len(key_params)}个)", "自定义"],
            key="auto_param_preset"
        )

        if param_preset.startswith("全参数"):
            param_spaces = all_params
        elif param_preset.startswith("关键参数"):
            param_spaces = key_params
        else:
            # 自定义参数选择
            param_groups = ParamSpaceManager.get_param_groups(strategy_name)
            selected_params = []

            st.write("**选择要优化的参数:**")
            for group_name, param_names in param_groups.items():
                with st.expander(group_name, expanded=True):
                    for pname in param_names:
                        if pname in all_params:
                            space = all_params[pname]
                            label = space.label or pname
                            if st.checkbox(f"{label} ({pname})", value=True, key=f"custom_{pname}"):
                                selected_params.append(pname)

            param_spaces = {k: v for k, v in all_params.items() if k in selected_params}

        st.info(f"已选择 {len(param_spaces)} 个参数")

        # 显示参数范围预览
        if param_spaces:
            with st.expander("参数范围预览"):
                for name, space in list(param_spaces.items())[:5]:
                    label = space.label or name
                    st.caption(f"{label}: {space.low} ~ {space.high}")
                if len(param_spaces) > 5:
                    st.caption(f"... 还有 {len(param_spaces) - 5} 个参数")

    with col3:
        st.markdown("#### 优化设置")

        # 时间范围
        st.write("**训练集**")
        train_col1, train_col2 = st.columns(2)
        with train_col1:
            train_start = st.date_input("开始", datetime(2019, 1, 1), key="auto_train_start")
        with train_col2:
            train_end = st.date_input("结束", datetime(2023, 12, 31), key="auto_train_end")

        st.write("**验证集**")
        val_col1, val_col2 = st.columns(2)
        with val_col1:
            val_start = st.date_input("开始", datetime(2024, 1, 1), key="auto_val_start")
        with val_col2:
            val_end = st.date_input("结束", datetime.now(), key="auto_val_end")

        # 时间周期选择
        timeframe = st.selectbox(
            "K线周期",
            ["1h", "4h", "1d"],
            index=0,
            format_func=lambda x: {"1h": "1小时", "4h": "4小时", "1d": "日线"}[x],
            key="auto_timeframe"
        )

        # 优化轮数
        n_trials = st.slider("优化轮数", 20, 200, 50, 10, key="auto_n_trials")

        # 优化目标
        objective = st.selectbox(
            "优化目标",
            ["sharpe", "calmar", "return", "sortino"],
            format_func=lambda x: {"sharpe": "夏普比率", "calmar": "卡玛比率", "return": "总收益率", "sortino": "Sortino比率"}[x],
            key="auto_objective"
        )

        # 高级设置
        with st.expander("高级设置"):
            initial_capital = st.number_input("初始资金", 50000, 1000000, 100000, 10000, key="auto_capital")
            min_trades = st.number_input("最少交易次数", 1, 50, 5, 1, key="auto_min_trades")
            max_drawdown = st.slider("最大回撤限制", 0.20, 0.60, 0.40, 0.05, key="auto_max_dd")

    st.markdown("---")

    # 一键启动按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_btn = st.button(
            "开始优化并保存配置",
            type="primary",
            use_container_width=True,
            key="auto_run_btn"
        )

    if run_btn:
        if not selected_symbols:
            st.error("请至少选择一个品种")
            return

        if not param_spaces:
            st.error("请至少选择一个参数")
            return

        # 时间范围校验
        if train_start >= train_end:
            st.error("训练集开始日期必须早于结束日期")
            return
        if val_start >= val_end:
            st.error("验证集开始日期必须早于结束日期")
            return
        if train_end >= val_start:
            st.error("训练集结束日期必须早于验证集开始日期（避免数据泄露）")
            return

        # 检查时间跨度是否合理
        train_days = (train_end - train_start).days
        val_days = (val_end - val_start).days
        if train_days < 180:
            st.warning(f"训练集仅{train_days}天，建议至少180天以获得稳健参数")
        if val_days < 60:
            st.warning(f"验证集仅{val_days}天，建议至少60天以检验过拟合")

        # 创建优化配置
        config = OptimizationConfig(
            strategy_name=strategy_name,
            symbols=selected_symbols,
            train_start=train_start.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            val_start=val_start.strftime("%Y-%m-%d"),
            val_end=val_end.strftime("%Y-%m-%d"),
            timeframe=timeframe,
            n_trials=n_trials,
            objective=objective,
            initial_capital=initial_capital,
            min_trades=min_trades,
            max_drawdown=max_drawdown,
            per_symbol=(opt_mode == "每品种独立优化")
        )

        # 执行优化
        _run_auto_optimization(config, param_spaces)

    # 显示历史记录
    st.markdown("---")
    _show_optimization_history()


def _run_auto_optimization(config, param_spaces):
    """执行自动优化"""
    from optimization import OptunaOptimizer, ConfigApplier

    # 进度条和日志区域
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.container()

    logs = []

    def log_callback(msg):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        with log_container:
            st.code("\n".join(logs[-15:]))

    def progress_callback(progress, msg):
        progress_bar.progress(min(progress, 1.0))
        status_text.text(msg)

    # 创建优化器
    optimizer = OptunaOptimizer(config)
    optimizer.set_log_callback(log_callback)
    optimizer.set_progress_callback(progress_callback)

    try:
        # 执行优化
        if config.per_symbol:
            # 每品种独立优化
            results = optimizer.optimize_per_symbol(param_spaces)
            _display_multi_results(results, config)

            # 批量保存配置
            saved_files = ConfigApplier.save_all_symbols(results)
            st.success(f"已保存 {len(saved_files)} 个配置文件")

            for symbol, filepath in saved_files.items():
                st.info(f"{symbol}: {os.path.basename(filepath)}")

            # 保存到session_state供其他页面使用
            st.session_state['opt_results'] = {
                symbol: ConfigApplier.apply_to_session_state(result, symbol)
                for symbol, result in results.items()
            }

        else:
            # 多品种综合优化
            result = optimizer.optimize(param_spaces)
            _display_single_result(result, config)

            # 保存配置
            filepath = ConfigApplier.save_optimized_config(result, symbol=None)
            st.success(f"配置已保存: {os.path.basename(filepath)}")

            # 保存到session_state
            st.session_state['opt_result'] = ConfigApplier.apply_to_session_state(result)

    except Exception as e:
        st.error(f"优化失败: {e}")
        logger.exception("优化失败")


def _display_single_result(result, config):
    """显示单个优化结果"""
    st.markdown("---")
    st.markdown("### 优化结果")

    # 最优参数
    st.markdown("#### 最优参数")
    params_df = pd.DataFrame([
        {"参数": k, "最优值": f"{v:.4f}" if isinstance(v, float) else str(v)}
        for k, v in result.best_params.items()
    ])
    st.dataframe(params_df, hide_index=True, use_container_width=True)

    # 性能对比
    st.markdown("#### 性能指标")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("训练集Sharpe", f"{result.train_metrics.get('sharpe', 0):.3f}")
        st.metric("训练集收益", f"{result.train_metrics.get('return', 0) * 100:.1f}%")

    with col2:
        st.metric("验证集Sharpe", f"{result.val_metrics.get('sharpe', 0):.3f}")
        st.metric("验证集收益", f"{result.val_metrics.get('return', 0) * 100:.1f}%")

    with col3:
        train_sharpe = result.train_metrics.get('sharpe', 0)
        val_sharpe = result.val_metrics.get('sharpe', 0)
        decay = (train_sharpe - val_sharpe) / train_sharpe * 100 if train_sharpe > 0 else 0

        if decay > OVERFITTING_THRESHOLD_HIGH:
            st.error(f"衰减: {decay:.1f}%")
            st.caption("过拟合风险高")
        elif decay > OVERFITTING_THRESHOLD_MEDIUM:
            st.warning(f"衰减: {decay:.1f}%")
            st.caption("轻度过拟合")
        else:
            st.success(f"衰减: {decay:.1f}%")
            st.caption("参数稳健")

    # 参数重要性
    if result.param_importance:
        st.markdown("#### 参数重要性")

        imp_df = pd.DataFrame([
            {"参数": k, "重要性": v}
            for k, v in sorted(result.param_importance.items(), key=lambda x: -x[1])
        ])

        fig = go.Figure(go.Bar(
            x=imp_df['重要性'],
            y=imp_df['参数'],
            orientation='h',
            marker_color='#1f77b4'
        ))
        fig.update_layout(
            height=max(200, len(imp_df) * 25),
            margin=dict(l=120, r=50, t=30, b=30),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)

    # 优化收敛图
    if not result.optimization_history.empty:
        st.markdown("#### 优化收敛过程")

        history = result.optimization_history
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=history['trial'],
            y=history['sharpe'],
            mode='markers',
            name='试验值',
            marker=dict(size=6, color='#1f77b4', opacity=0.6)
        ))

        # 添加累计最优线
        cummax = history['sharpe'].cummax()
        fig.add_trace(go.Scatter(
            x=history['trial'],
            y=cummax,
            mode='lines',
            name='累计最优',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            height=300,
            xaxis_title='Trial',
            yaxis_title='Sharpe',
            margin=dict(l=50, r=50, t=30, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)


def _display_multi_results(results: dict, config):
    """显示多品种优化结果"""
    st.markdown("---")
    st.markdown("### 优化结果")

    # 汇总表
    summary_data = []
    for symbol, result in results.items():
        summary_data.append({
            "品种": symbol,
            f"最优{config.objective}": f"{result.best_value:.3f}",
            "训练Sharpe": f"{result.train_metrics.get('sharpe', 0):.3f}",
            "验证Sharpe": f"{result.val_metrics.get('sharpe', 0):.3f}",
            "验证收益": f"{result.val_metrics.get('return', 0) * 100:.1f}%",
            "验证回撤": f"{result.val_metrics.get('max_drawdown', 0) * 100:.1f}%",
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # 详细参数（可展开）
    with st.expander("查看各品种最优参数"):
        for symbol, result in results.items():
            st.markdown(f"**{symbol}**")
            params_df = pd.DataFrame([
                {"参数": k, "值": f"{v:.4f}" if isinstance(v, float) else str(v)}
                for k, v in result.best_params.items()
            ])
            st.dataframe(params_df, hide_index=True, use_container_width=True)
            st.markdown("---")


def _show_optimization_history():
    """显示优化历史"""
    st.markdown("### 优化历史")

    try:
        from optimization import ConfigApplier
        history = ConfigApplier.get_optimization_history()

        if not history:
            st.info("暂无优化历史记录")
            return

        # 转换为DataFrame
        history_df = pd.DataFrame(history)

        # 格式化显示
        display_df = history_df[['strategy', 'symbol', 'best_value', 'train_sharpe', 'val_sharpe', 'optimized_at']].copy()
        display_df.columns = ['策略', '品种', '最优值', '训练Sharpe', '验证Sharpe', '优化时间']

        st.dataframe(display_df, hide_index=True, use_container_width=True)

        # 选择应用
        if len(history) > 0:
            st.markdown("#### 应用历史配置")
            selected_config = st.selectbox(
                "选择配置",
                history_df['config_file'].tolist(),
                format_func=lambda x: f"{x.replace('.yml', '')}",
                key="select_history_config"
            )

            if st.button("应用到回测", key="apply_history_btn"):
                from config_manager import load_config
                config = load_config(selected_config)
                if config and 'run_policy' in config:
                    st.session_state['opt_result'] = {
                        'strategy_name': config['run_policy'].get('name', ''),
                        'params': config['run_policy'].get('params', {}),
                        'config_file': selected_config
                    }
                    st.success(f"已应用配置: {selected_config}")
                    st.info("请前往回测页面查看应用的参数")

    except Exception as e:
        st.warning(f"加载历史记录失败: {e}")


# 独立运行入口
if __name__ == "__main__":
    st.set_page_config(page_title="AI参数优化", layout="wide")
    render_auto_optimizer_page()
